/*
 * Copyright 2023, 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>

#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/util/Worklist.hpp>

#include <limits>
#include <queue>

namespace jlm::llvm::aa
{

/**
 * Flag that enables unification logic.
 * When enabled, each points-to set lookup needs to perform a find operation.
 * When disabled, attempting to call UnifyPointerObjects panics.
 */
static constexpr bool ENABLE_UNIFICATION = true;

PointerObjectIndex
PointerObjectSet::AddPointerObject(PointerObjectKind kind)
{
  JLM_ASSERT(PointerObjects_.size() < std::numeric_limits<PointerObjectIndex>::max());
  PointerObjectIndex index = PointerObjects_.size();

  PointerObjects_.emplace_back(kind);
  if constexpr (ENABLE_UNIFICATION)
  {
    PointerObjectParents_.push_back(index);
    PointerObjectRank_.push_back(0);
  }
  PointsToSets_.emplace_back(); // Add empty points-to set
  return PointerObjects_.size() - 1;
}

size_t
PointerObjectSet::NumPointerObjects() const noexcept
{
  return PointerObjects_.size();
}

size_t
PointerObjectSet::NumPointerObjectsOfKind(PointerObjectKind kind) const noexcept
{
  size_t count = 0;
  for (auto & pointerObject : PointerObjects_)
  {
    count += pointerObject.Kind == kind;
  }
  return count;
}

PointerObjectIndex
PointerObjectSet::CreateRegisterPointerObject(const rvsdg::output & rvsdgOutput)
{
  JLM_ASSERT(RegisterMap_.count(&rvsdgOutput) == 0);
  return RegisterMap_[&rvsdgOutput] = AddPointerObject(PointerObjectKind::Register);
}

PointerObjectIndex
PointerObjectSet::GetRegisterPointerObject(const rvsdg::output & rvsdgOutput) const
{
  const auto it = RegisterMap_.find(&rvsdgOutput);
  JLM_ASSERT(it != RegisterMap_.end());
  return it->second;
}

std::optional<PointerObjectIndex>
PointerObjectSet::TryGetRegisterPointerObject(const rvsdg::output & rvsdgOutput) const
{
  if (const auto it = RegisterMap_.find(&rvsdgOutput); it != RegisterMap_.end())
    return it->second;
  return std::nullopt;
}

void
PointerObjectSet::MapRegisterToExistingPointerObject(
    const rvsdg::output & rvsdgOutput,
    PointerObjectIndex pointerObject)
{
  JLM_ASSERT(RegisterMap_.count(&rvsdgOutput) == 0);
  JLM_ASSERT(GetPointerObjectKind(pointerObject) == PointerObjectKind::Register);
  RegisterMap_[&rvsdgOutput] = pointerObject;
}

PointerObjectIndex
PointerObjectSet::CreateDummyRegisterPointerObject()
{
  return AddPointerObject(PointerObjectKind::Register);
}

PointerObjectIndex
PointerObjectSet::CreateAllocaMemoryObject(const rvsdg::node & allocaNode)
{
  JLM_ASSERT(AllocaMap_.count(&allocaNode) == 0);
  return AllocaMap_[&allocaNode] = AddPointerObject(PointerObjectKind::AllocaMemoryObject);
}

PointerObjectIndex
PointerObjectSet::CreateMallocMemoryObject(const rvsdg::node & mallocNode)
{
  JLM_ASSERT(MallocMap_.count(&mallocNode) == 0);
  return MallocMap_[&mallocNode] = AddPointerObject(PointerObjectKind::MallocMemoryObject);
}

PointerObjectIndex
PointerObjectSet::CreateGlobalMemoryObject(const delta::node & deltaNode)
{
  JLM_ASSERT(GlobalMap_.count(&deltaNode) == 0);
  return GlobalMap_[&deltaNode] = AddPointerObject(PointerObjectKind::GlobalMemoryObject);
}

PointerObjectIndex
PointerObjectSet::CreateFunctionMemoryObject(const lambda::node & lambdaNode)
{
  JLM_ASSERT(!FunctionMap_.HasKey(&lambdaNode));
  const auto pointerObject = AddPointerObject(PointerObjectKind::FunctionMemoryObject);
  FunctionMap_.Insert(&lambdaNode, pointerObject);
  return pointerObject;
}

PointerObjectIndex
PointerObjectSet::GetFunctionMemoryObject(const lambda::node & lambdaNode) const
{
  JLM_ASSERT(FunctionMap_.HasKey(&lambdaNode));
  return FunctionMap_.LookupKey(&lambdaNode);
}

const lambda::node &
PointerObjectSet::GetLambdaNodeFromFunctionMemoryObject(PointerObjectIndex index) const
{
  JLM_ASSERT(FunctionMap_.HasValue(index));
  return *FunctionMap_.LookupValue(index);
}

PointerObjectIndex
PointerObjectSet::CreateImportMemoryObject(const rvsdg::argument & importNode)
{
  JLM_ASSERT(ImportMap_.count(&importNode) == 0);
  auto importMemoryObject = AddPointerObject(PointerObjectKind::ImportMemoryObject);
  ImportMap_[&importNode] = importMemoryObject;

  // Memory objects defined in other modules are definitely not private to this module
  MarkAsEscaped(importMemoryObject);
  return importMemoryObject;
}

const std::unordered_map<const rvsdg::output *, PointerObjectIndex> &
PointerObjectSet::GetRegisterMap() const noexcept
{
  return RegisterMap_;
}

const std::unordered_map<const rvsdg::node *, PointerObjectIndex> &
PointerObjectSet::GetAllocaMap() const noexcept
{
  return AllocaMap_;
}

const std::unordered_map<const rvsdg::node *, PointerObjectIndex> &
PointerObjectSet::GetMallocMap() const noexcept
{
  return MallocMap_;
}

const std::unordered_map<const delta::node *, PointerObjectIndex> &
PointerObjectSet::GetGlobalMap() const noexcept
{
  return GlobalMap_;
}

const util::BijectiveMap<const lambda::node *, PointerObjectIndex> &
PointerObjectSet::GetFunctionMap() const noexcept
{
  return FunctionMap_;
}

const std::unordered_map<const rvsdg::argument *, PointerObjectIndex> &
PointerObjectSet::GetImportMap() const noexcept
{
  return ImportMap_;
}

PointerObjectKind
PointerObjectSet::GetPointerObjectKind(PointerObjectIndex index) const noexcept
{
  JLM_ASSERT(index < NumPointerObjects());
  return PointerObjects_[index].Kind;
}

bool
PointerObjectSet::ShouldTrackPointees(PointerObjectIndex index) const noexcept
{
  JLM_ASSERT(index < NumPointerObjects());
  return PointerObjects_[index].ShouldTrackPointees();
}

bool
PointerObjectSet::IsPointerObjectRegister(PointerObjectIndex index) const noexcept
{
  JLM_ASSERT(index < NumPointerObjects());
  return PointerObjects_[index].IsRegister();
}

bool
PointerObjectSet::HasEscaped(PointerObjectIndex index) const noexcept
{
  JLM_ASSERT(index < NumPointerObjects());
  return PointerObjects_[index].HasEscaped;
}

bool
PointerObjectSet::MarkAsEscaped(PointerObjectIndex index)
{
  // Registers do not have addresses, and can as such not escape
  JLM_ASSERT(!IsPointerObjectRegister(index));
  if (PointerObjects_[index].HasEscaped)
    return false;

  PointerObjects_[index].HasEscaped = true;

  // Flags implied by escaping
  MarkAsPointeesEscaping(index);
  MarkAsPointingToExternal(index);

  return true;
}

[[nodiscard]] bool
PointerObjectSet::HasPointeesEscaping(PointerObjectIndex index) const noexcept
{
  return PointerObjects_[GetUnificationRoot(index)].PointeesEscaping;
}

bool
PointerObjectSet::MarkAsPointeesEscaping(PointerObjectIndex index)
{
  auto root = GetUnificationRoot(index);
  if (PointerObjects_[root].PointeesEscaping)
    return false;
  PointerObjects_[root].PointeesEscaping = true;
  return true;
}

bool
PointerObjectSet::IsPointingToExternal(PointerObjectIndex index) const noexcept
{
  return PointerObjects_[GetUnificationRoot(index)].PointsToExternal;
}

bool
PointerObjectSet::MarkAsPointingToExternal(PointerObjectIndex index)
{
  auto root = GetUnificationRoot(index);
  if (PointerObjects_[root].PointsToExternal)
    return false;
  PointerObjects_[root].PointsToExternal = true;
  return true;
}

PointerObjectIndex
PointerObjectSet::GetUnificationRoot(PointerObjectIndex index) const noexcept
{
  JLM_ASSERT(index < NumPointerObjects());

  if constexpr (ENABLE_UNIFICATION)
  {
    // Technique known as path halving, gives same asymptotic performance as full path compression
    while (PointerObjectParents_[index] != index)
    {
      auto & parent = PointerObjectParents_[index];
      auto grandparent = PointerObjectParents_[parent];
      index = parent = grandparent;
    }
  }

  return index;
}

bool
PointerObjectSet::IsUnificationRoot(PointerObjectIndex index) const noexcept
{
  return GetUnificationRoot(index) == index;
}

PointerObjectIndex
PointerObjectSet::UnifyPointerObjects(PointerObjectIndex object1, PointerObjectIndex object2)
{
  if constexpr (!ENABLE_UNIFICATION)
    JLM_UNREACHABLE("Unification is not enabled");

  PointerObjectIndex newRoot = GetUnificationRoot(object1);
  PointerObjectIndex oldRoot = GetUnificationRoot(object2);

  if (newRoot == oldRoot)
    return newRoot;

  // Make sure the rank continues to be an upper bound for height.
  // If they have different rank, the root should be the one with the highest rank.
  // Equal rank forces the new root to increase its rank
  if (PointerObjectRank_[newRoot] < PointerObjectRank_[oldRoot])
    std::swap(newRoot, oldRoot);
  else if (PointerObjectRank_[newRoot] == PointerObjectRank_[oldRoot])
    PointerObjectRank_[newRoot]++;

  PointerObjectParents_[oldRoot] = newRoot;

  PointsToSets_[newRoot].UnionWith(PointsToSets_[oldRoot]);
  PointsToSets_[oldRoot].Clear();

  // Ensure any flags set on the points-to set continue to be set in the new unification
  if (IsPointingToExternal(oldRoot))
    MarkAsPointingToExternal(newRoot);
  if (HasPointeesEscaping(oldRoot))
    MarkAsPointeesEscaping(newRoot);

  return newRoot;
}

const util::HashSet<PointerObjectIndex> &
PointerObjectSet::GetPointsToSet(PointerObjectIndex index) const
{
  return PointsToSets_[GetUnificationRoot(index)];
}

// Makes pointee a member of P(pointer)
bool
PointerObjectSet::AddToPointsToSet(PointerObjectIndex pointer, PointerObjectIndex pointee)
{
  JLM_ASSERT(pointer < NumPointerObjects());
  JLM_ASSERT(pointee < NumPointerObjects());
  // Assert we are not trying to point to a register
  JLM_ASSERT(!IsPointerObjectRegister(pointee));

  const auto pointerRoot = GetUnificationRoot(pointer);

  return PointsToSets_[pointerRoot].Insert(pointee);
}

// Makes P(superset) a superset of P(subset)
bool
PointerObjectSet::MakePointsToSetSuperset(PointerObjectIndex superset, PointerObjectIndex subset)
{
  auto supersetRoot = GetUnificationRoot(superset);
  auto subsetRoot = GetUnificationRoot(subset);

  if (supersetRoot == subsetRoot)
    return false;

  auto & P_super = PointsToSets_[supersetRoot];
  auto & P_sub = PointsToSets_[subsetRoot];

  bool modified = P_super.UnionWith(P_sub);

  // If the external node is in the subset, it must also be part of the superset
  if (IsPointingToExternal(subsetRoot))
    modified |= MarkAsPointingToExternal(supersetRoot);

  return modified;
}

std::unique_ptr<PointerObjectSet>
PointerObjectSet::Clone() const
{
  return std::make_unique<PointerObjectSet>(*this);
}

// Makes P(superset) a superset of P(subset)
bool
SupersetConstraint::ApplyDirectly(PointerObjectSet & set)
{
  return set.MakePointsToSetSuperset(Superset_, Subset_);
}

// for all x in P(pointer1), make P(x) a superset of P(pointer2)
bool
StoreConstraint::ApplyDirectly(PointerObjectSet & set)
{
  bool modified = false;
  for (PointerObjectIndex x : set.GetPointsToSet(Pointer_).Items())
    modified |= set.MakePointsToSetSuperset(x, Value_);

  // If external in P(pointer), P(external) should become a superset of P(value)
  // In practice, this means everything in P(value) escapes
  if (set.IsPointingToExternal(Pointer_))
    modified |= set.MarkAsPointeesEscaping(Value_);

  return modified;
}

// Make P(loaded) a superset of P(x) for all x in P(pointer)
bool
LoadConstraint::ApplyDirectly(PointerObjectSet & set)
{
  bool modified = false;
  for (PointerObjectIndex x : set.GetPointsToSet(Pointer_).Items())
    modified |= set.MakePointsToSetSuperset(Value_, x);

  // P(pointer) "contains" external, then P(loaded) should also "contain" it
  if (set.IsPointingToExternal(Pointer_))
    modified |= set.MarkAsPointingToExternal(Value_);

  return modified;
}

/**
 * Handles informing the arguments and return values of the CallNode about
 * possibly being sent to and retrieved from unknown code.
 * @param set the PointerObjectSet representing this module.
 * @param callNode the RVSDG CallNode that represents the function call itself
 * @param markAsPointeesEscaping the function to call when marking a register as pointees escaping
 * @param markAsPointsToExternal called to flag a PointerObject as pointing to external
 */
template<typename MarkAsPointeesEscaping, typename MarkAsPointsToExternal>
void
HandleCallingExternalFunction(
    PointerObjectSet & set,
    const jlm::llvm::CallNode & callNode,
    MarkAsPointeesEscaping & markAsPointeesEscaping,
    MarkAsPointsToExternal & markAsPointsToExternal)
{

  // Mark all the call's inputs as escaped, and all the outputs as pointing to external
  for (size_t n = 0; n < callNode.NumArguments(); n++)
  {
    const auto & inputRegister = *callNode.Argument(n)->origin();
    const auto inputRegisterPO = set.TryGetRegisterPointerObject(inputRegister);

    if (inputRegisterPO)
      markAsPointeesEscaping(inputRegisterPO.value());
  }

  for (size_t n = 0; n < callNode.NumResults(); n++)
  {
    const auto & outputRegister = *callNode.Result(n);
    const auto outputRegisterPO = set.TryGetRegisterPointerObject(outputRegister);
    if (outputRegisterPO)
      markAsPointsToExternal(outputRegisterPO.value());
  }
}

/**
 * Handles informing the arguments and return values of the CallNode about
 * possibly being sent to and received from a given PointerObject of ImportMemoryObject.
 * @param set the PointerObjectSet representing this module.
 * @param callNode the RVSDG CallNode that represents the function call itself
 * @param imported the PointerObject of ImportMemoryObject kind that might be called.
 * @param markAsPointeesEscaping the function to call when marking a register as pointees escaping
 * @param markAsPointsToExternal called to flag a PointerObject as pointing to external
 */
template<typename MarkAsPointeesEscaping, typename MarkAsPointsToExternal>
static void
HandleCallingImportedFunction(
    PointerObjectSet & set,
    const jlm::llvm::CallNode & callNode,
    [[maybe_unused]] PointerObjectIndex imported,
    MarkAsPointeesEscaping & markAsPointeesEscaping,
    MarkAsPointsToExternal & markAsPointsToExternal)
{
  // FIXME: Add special handling of common library functions
  // Otherwise we don't know anything about the function
  return HandleCallingExternalFunction(
      set,
      callNode,
      markAsPointeesEscaping,
      markAsPointsToExternal);
}

/**
 * Pairs up call node inputs and function body argument, and calls the provided \p makeSuperset
 * to make the function body arguments include all pointees the call node may pass in.
 */
template<typename MakeSupersetFunctor>
static void
HandleLambdaCallParameters(
    PointerObjectSet & set,
    const jlm::llvm::CallNode & callNode,
    const lambda::node & lambdaNode,
    MakeSupersetFunctor & makeSuperset)
{
  for (size_t n = 0; n < callNode.NumArguments() && n < lambdaNode.nfctarguments(); n++)
  {
    const auto & inputRegister = *callNode.Argument(n)->origin();
    const auto & argumentRegister = *lambdaNode.fctargument(n);

    const auto inputRegisterPO = set.TryGetRegisterPointerObject(inputRegister);
    const auto argumentRegisterPO = set.TryGetRegisterPointerObject(argumentRegister);
    if (!inputRegisterPO || !argumentRegisterPO)
      continue;

    makeSuperset(*argumentRegisterPO, *inputRegisterPO);
  }
}

/**
 * Pairs up function body results and call node outputs, and calls the provided \p makeSuperset
 * to make the call node output include all possible returned pointees.
 */
template<typename MakeSupersetFunctor>
static void
HandleLambdaCallReturnValues(
    PointerObjectSet & set,
    const jlm::llvm::CallNode & callNode,
    const lambda::node & lambdaNode,
    MakeSupersetFunctor & makeSuperset)
{
  for (size_t n = 0; n < callNode.NumResults() && n < lambdaNode.nfctresults(); n++)
  {
    const auto & outputRegister = *callNode.Result(n);
    const auto & resultRegister = *lambdaNode.fctresult(n)->origin();

    const auto outputRegisterPO = set.TryGetRegisterPointerObject(outputRegister);
    const auto resultRegisterPO = set.TryGetRegisterPointerObject(resultRegister);
    if (!outputRegisterPO || !resultRegisterPO)
      continue;

    makeSuperset(*outputRegisterPO, *resultRegisterPO);
  }
}

/**
 * Handles informing the CallNode about possibly calling the function represented by \p lambda.
 * Passes the points-to-sets of the arguments into the function subregion,
 * and passes the points-to-sets of the function's return values back to the CallNode's outputs.
 * Passing pointees is performed by calling the provided \p makeSuperset functor, with signature
 *   void(PointerObjectIndex superset, PointerObjectIndex subset)
 * @param set the PointerObjectSet representing this module.
 * @param callNode the RVSDG CallNode that represents the function call itself
 * @param lambda the PointerObject of FunctionMemoryObject kind that might be called.
 * @param makeSuperset the function to call to make one points-to set a superset of another
 */
template<typename MakeSupersetFunctor>
static void
HandleCallingLambdaFunction(
    PointerObjectSet & set,
    const jlm::llvm::CallNode & callNode,
    PointerObjectIndex lambda,
    MakeSupersetFunctor & makeSuperset)
{
  auto & lambdaNode = set.GetLambdaNodeFromFunctionMemoryObject(lambda);

  // LLVM allows calling functions even when the number of arguments don't match,
  // so we instead pair up as many parameters and return values as possible

  // Pass all call node inputs to the function's subregion
  HandleLambdaCallParameters(set, callNode, lambdaNode, makeSuperset);

  // Pass the function's subregion results to the output of the call node
  HandleLambdaCallReturnValues(set, callNode, lambdaNode, makeSuperset);
}

// Connects function calls to every possible target function
bool
FunctionCallConstraint::ApplyDirectly(PointerObjectSet & set)
{
  bool modified = false;

  const auto MakeSuperset = [&](PointerObjectIndex superset, PointerObjectIndex subset)
  {
    modified |= set.MakePointsToSetSuperset(superset, subset);
  };

  const auto MarkAsPointeesEscaping = [&](PointerObjectIndex index)
  {
    modified |= set.MarkAsPointeesEscaping(index);
  };

  const auto MarkAsPointsToExternal = [&](PointerObjectIndex index)
  {
    modified |= set.MarkAsPointingToExternal(index);
  };

  // For each possible function target, connect parameters and return values to the call node
  for (const auto target : set.GetPointsToSet(Pointer_).Items())
  {
    const auto kind = set.GetPointerObjectKind(target);
    if (kind == PointerObjectKind::ImportMemoryObject)
      HandleCallingImportedFunction(
          set,
          CallNode_,
          target,
          MarkAsPointeesEscaping,
          MarkAsPointsToExternal);
    else if (kind == PointerObjectKind::FunctionMemoryObject)
      HandleCallingLambdaFunction(set, CallNode_, target, MakeSuperset);
  }

  // If we might be calling an external function
  if (set.IsPointingToExternal(Pointer_))
    HandleCallingExternalFunction(set, CallNode_, MarkAsPointeesEscaping, MarkAsPointsToExternal);

  return modified;
}

bool
EscapeFlagConstraint::PropagateEscapedFlagsDirectly(PointerObjectSet & set)
{
  std::queue<PointerObjectIndex> pointeeEscapers;

  // First add all unification roots marked as PointeesEscaping
  for (PointerObjectIndex idx = 0; idx < set.NumPointerObjects(); idx++)
  {
    if (set.IsUnificationRoot(idx) && set.HasPointeesEscaping(idx))
      pointeeEscapers.push(idx);
  }

  bool modified = false;

  // For all pointee escapers, check if they point to any PointerObjects not marked as escaped
  while (!pointeeEscapers.empty())
  {
    const PointerObjectIndex pointeeEscaper = pointeeEscapers.front();
    pointeeEscapers.pop();

    for (PointerObjectIndex pointee : set.GetPointsToSet(pointeeEscaper).Items())
    {
      const auto unificationRoot = set.GetUnificationRoot(pointee);
      const bool prevHasPointeesEscaping = set.HasPointeesEscaping(unificationRoot);

      modified |= set.MarkAsEscaped(pointee);

      // If the pointee's unification root previously didn't have the PointeesEscaping flag,
      // add it to the queue
      if (!prevHasPointeesEscaping)
      {
        JLM_ASSERT(set.HasPointeesEscaping(unificationRoot));
        pointeeEscapers.push(unificationRoot);
      }
    }
  }

  return modified;
}

/**
 * Given an escaped function, the results registers should be marked as escaping pointees,
 * and all arguments as pointing to external, provided they are of types we track the pointees of.
 * The modifications are made using the provided functors, which are called only if any flags are
 * missing. Each functor takes a single parameter:
 *   The index of a PointerObject of Register kind that is missing the specified flag.
 * @param set the PointerObjectSet representing this module
 * @param lambda the escaped PointerObject of function kind
 * @param markAsPointeesEscaping the function to call when marking a register as pointees escaping
 * @param markAsPointsToExternal called to flag a PointerObject as pointing to external
 */
template<typename MarkAsPointeesEscapingFunctor, typename MarkAsPointsToExternalFunctor>
static void
HandleEscapedFunction(
    PointerObjectSet & set,
    PointerObjectIndex lambda,
    MarkAsPointeesEscapingFunctor & markAsPointeesEscaping,
    MarkAsPointsToExternalFunctor & markAsPointsToExternal)
{
  JLM_ASSERT(set.GetPointerObjectKind(lambda) == PointerObjectKind::FunctionMemoryObject);
  JLM_ASSERT(set.HasEscaped(lambda));

  // We now go through the lambda's inner region and apply the necessary flags
  auto & lambdaNode = set.GetLambdaNodeFromFunctionMemoryObject(lambda);

  // All the function's arguments need to be flagged as PointsToExternal
  for (auto & argument : lambdaNode.fctarguments())
  {
    // Argument registers that are mapped to a register pointer object should point to external
    const auto argumentPO = set.TryGetRegisterPointerObject(argument);
    if (!argumentPO)
      continue;

    // Nothing to be done if it is already marked as points to external
    if (set.IsPointingToExternal(argumentPO.value()))
      continue;

    markAsPointsToExternal(argumentPO.value());
  }

  // All results of pointer type need to be flagged as HasEscaped
  for (auto & result : lambdaNode.fctresults())
  {
    const auto resultPO = set.TryGetRegisterPointerObject(*result.origin());
    if (!resultPO)
      continue;

    // Nothing to be done if it is already marked as escaped
    if (set.HasEscaped(resultPO.value()))
      continue;

    // Mark the result register as escaping any pointees it may have
    markAsPointeesEscaping(resultPO.value());
  }
}

bool
EscapedFunctionConstraint::PropagateEscapedFunctionsDirectly(PointerObjectSet & set)
{
  bool modified = false;

  const auto markAsPointeesEscaping = [&](PointerObjectIndex index)
  {
    modified |= set.MarkAsPointeesEscaping(index);
  };

  const auto markAsPointsToExternal = [&](PointerObjectIndex index)
  {
    modified |= set.MarkAsPointingToExternal(index);
  };

  for (const auto [lambda, lambdaPO] : set.GetFunctionMap())
  {
    if (set.HasEscaped(lambdaPO))
      HandleEscapedFunction(set, lambdaPO, markAsPointeesEscaping, markAsPointsToExternal);
  }

  return modified;
}

void
PointerObjectConstraintSet::AddPointerPointeeConstraint(
    PointerObjectIndex pointer,
    PointerObjectIndex pointee)
{
  // All set constraints are additive, so simple constraints like this can be directly applied.
  Set_.AddToPointsToSet(pointer, pointee);
}

void
PointerObjectConstraintSet::AddPointsToExternalConstraint(PointerObjectIndex pointer)
{
  // Flags are never removed, so adding the flag now ensures it will be included.
  Set_.MarkAsPointingToExternal(pointer);
}

void
PointerObjectConstraintSet::AddRegisterContentEscapedConstraint(PointerObjectIndex registerIndex)
{
  JLM_ASSERT(Set_.IsPointerObjectRegister(registerIndex));
  Set_.MarkAsPointeesEscaping(registerIndex);
}

void
PointerObjectConstraintSet::AddConstraint(ConstraintVariant c)
{
  Constraints_.push_back(c);
}

const std::vector<PointerObjectConstraintSet::ConstraintVariant> &
PointerObjectConstraintSet::GetConstraints() const noexcept
{
  return Constraints_;
}

size_t
PointerObjectConstraintSet::SolveUsingWorklist()
{
  // Create auxiliary superset graph.
  // All edges must have their tail be a unification root.
  // If supersetEdges[x] contains y, (x -> y), that means P(y) supseteq P(x)
  std::vector<util::HashSet<PointerObjectIndex>> supersetEdges(Set_.NumPointerObjects());

  // Create quick lookup tables for Load, Store and function call constraints.
  // Lookup is indexed by the constraint's pointer
  // The constraints need to be added to the unification root, as only unification roots
  // are allowed on the worklist.
  std::vector<util::HashSet<PointerObjectIndex>> storeConstraints(Set_.NumPointerObjects());
  std::vector<util::HashSet<PointerObjectIndex>> loadConstraints(Set_.NumPointerObjects());
  std::vector<std::vector<const jlm::llvm::CallNode *>> callConstraints(Set_.NumPointerObjects());

  for (const auto & constraint : Constraints_)
  {
    if (const auto * ssConstraint = std::get_if<SupersetConstraint>(&constraint))
    {
      // Superset constraints become edges in the superset graph
      auto superset = Set_.GetUnificationRoot(ssConstraint->GetSuperset());
      auto subset = Set_.GetUnificationRoot(ssConstraint->GetSubset());

      supersetEdges[subset].Insert(superset);
    }
    else if (const auto * storeConstraint = std::get_if<StoreConstraint>(&constraint))
    {
      auto pointer = Set_.GetUnificationRoot(storeConstraint->GetPointer());
      auto value = Set_.GetUnificationRoot(storeConstraint->GetValue());

      storeConstraints[pointer].Insert(value);
    }
    else if (const auto * loadConstraint = std::get_if<LoadConstraint>(&constraint))
    {
      auto pointer = Set_.GetUnificationRoot(loadConstraint->GetPointer());
      auto value = Set_.GetUnificationRoot(loadConstraint->GetValue());

      loadConstraints[pointer].Insert(value);
    }
    else if (const auto * callConstraint = std::get_if<FunctionCallConstraint>(&constraint))
    {
      auto pointer = Set_.GetUnificationRoot(callConstraint->GetPointer());
      const auto & callNode = callConstraint->GetCallNode();

      callConstraints[pointer].push_back(&callNode);
    }
  }

  // The worklist, initialized with every unification root
  util::LrfWorklist<PointerObjectIndex> worklist;
  for (PointerObjectIndex i = 0; i < Set_.NumPointerObjects(); i++)
  {
    if (Set_.IsUnificationRoot(i))
      worklist.PushWorkItem(i);
  }

  // Helper function for adding superset edges, propagating everything currently in the subset.
  // The superset's root is added to the work list if its points-to set or flags are changed.
  const auto AddSupersetEdge = [&](PointerObjectIndex superset, PointerObjectIndex subset)
  {
    superset = Set_.GetUnificationRoot(superset);
    subset = Set_.GetUnificationRoot(subset);

    // If the edge already exists, ignore
    if (!supersetEdges[subset].Insert(superset))
      return;

    // A new edge was added, propagate points to-sets
    if (!Set_.MakePointsToSetSuperset(superset, subset))
      return;

    // pointees or the points-to-external flag were propagated to the superset, add to the worklist
    worklist.PushWorkItem(superset);
  };

  // Helper function for marking a PointerObject such that all its pointees will escape
  const auto MarkAsPointeesEscaping = [&](PointerObjectIndex index)
  {
    index = Set_.GetUnificationRoot(index);
    if (Set_.MarkAsPointeesEscaping(index))
      worklist.PushWorkItem(index);
  };

  // Helper function for flagging a pointer as pointing to external. Adds to the worklist if changed
  const auto MarkAsPointsToExternal = [&](PointerObjectIndex index)
  {
    index = Set_.GetUnificationRoot(index);
    if (Set_.MarkAsPointingToExternal(index))
      worklist.PushWorkItem(index);
  };

  // Ensure that all functions that have already escaped have informed their arguments and results
  // The worklist will only inform functions if their HasEscaped flag changes
  EscapedFunctionConstraint::PropagateEscapedFunctionsDirectly(Set_);

  // Count of the total number of work items fired
  size_t numWorkItems = 0;

  // The main worklist loop. A work item can be in the worklist for the following reasons:
  // - It has never been fired
  // - It has pointees added since the last time it was fired
  // - It has been marked as pointing to external since last time it was fired
  // - It has been marked as escaping all pointees since last time it was fired
  // All work items are unification roots, or were unification roots when added
  while (worklist.HasMoreWorkItems())
  {
    const auto n = worklist.PopWorkItem();
    numWorkItems++;

    // Only visit unification roots
    const auto root = Set_.GetUnificationRoot(n);
    if (n != root)
    {
      worklist.PushWorkItem(root);
      continue;
    }

    // Stores on the form *n = value.
    for (const auto value : storeConstraints[n].Items())
    {
      // This loop ensures *P(n) supseteq P(value)
      for (const auto pointee : Set_.GetPointsToSet(n).Items())
        AddSupersetEdge(pointee, value);

      // If P(n) contains "external", the contents of the written value escapes
      if (Set_.IsPointingToExternal(n))
        MarkAsPointeesEscaping(value);
    }

    // Loads on the form value = *n.
    for (const auto value : loadConstraints[n].Items())
    {
      // This loop ensures P(value) supseteq *P(n)
      for (const auto pointee : Set_.GetPointsToSet(n).Items())
        AddSupersetEdge(value, pointee);

      // If P(n) contains "external", the loaded value may also point to external
      if (Set_.IsPointingToExternal(n))
        MarkAsPointsToExternal(value);
    }

    // Function calls on the form (*n)()
    for (const auto callNode : callConstraints[n])
    {
      // Connect the inputs and outputs of the callNode to every possible function pointee
      for (const auto pointee : Set_.GetPointsToSet(n).Items())
      {
        const auto kind = Set_.GetPointerObjectKind(pointee);
        if (kind == PointerObjectKind::ImportMemoryObject)
          HandleCallingImportedFunction(
              Set_,
              *callNode,
              pointee,
              MarkAsPointeesEscaping,
              MarkAsPointsToExternal);
        else if (kind == PointerObjectKind::FunctionMemoryObject)
          HandleCallingLambdaFunction(Set_, *callNode, pointee, AddSupersetEdge);
      }

      // If P(n) contains "external", handle calling external functions
      if (Set_.IsPointingToExternal(n))
        HandleCallingExternalFunction(
            Set_,
            *callNode,
            MarkAsPointeesEscaping,
            MarkAsPointsToExternal);
    }

    // Propagate P(n) along all edges n -> superset
    for (auto superset : supersetEdges[n].Items())
    {
      // FIXME: Replace edges going to non-roots
      superset = Set_.GetUnificationRoot(superset);
      if (Set_.MakePointsToSetSuperset(superset, n))
        worklist.PushWorkItem(superset);
    }

    // If n is marked as PointeesEscaping, add the escaped flag to all pointees
    if (Set_.HasPointeesEscaping(n))
    {
      for (const auto pointee : Set_.GetPointsToSet(n).Items())
      {
        const auto pointeeRoot = Set_.GetUnificationRoot(pointee);
        const bool prevPointeesEscaping = Set_.HasPointeesEscaping(pointeeRoot);

        // Mark the pointee itself as escaped, not the pointee's unifiction root!
        if (!Set_.MarkAsEscaped(pointee))
          continue;

        // If the PointerObject we just marked as escaped is a function, inform it about escaping
        if (Set_.GetPointerObjectKind(pointee) == PointerObjectKind::FunctionMemoryObject)
          HandleEscapedFunction(Set_, pointee, MarkAsPointeesEscaping, MarkAsPointsToExternal);

        // If the pointee's unification root previously didn't have the PointeesEscaping flag,
        // add the unification root to the worklist
        if (!prevPointeesEscaping)
        {
          JLM_ASSERT(Set_.HasPointeesEscaping(pointeeRoot));
          worklist.PushWorkItem(pointeeRoot);
        }
      }
    }
  }

  return numWorkItems;
}

size_t
PointerObjectConstraintSet::SolveNaively()
{
  size_t numIterations = 0;

  // Keep applying constraints until no sets are modified
  bool modified = true;

  while (modified)
  {
    numIterations++;
    modified = false;

    for (auto & constraint : Constraints_)
    {
      std::visit(
          [&](auto & constraint)
          {
            modified |= constraint.ApplyDirectly(Set_);
          },
          constraint);
    }

    modified |= EscapeFlagConstraint::PropagateEscapedFlagsDirectly(Set_);
    modified |= EscapedFunctionConstraint::PropagateEscapedFunctionsDirectly(Set_);
  }

  return numIterations;
}

std::pair<std::unique_ptr<PointerObjectSet>, std::unique_ptr<PointerObjectConstraintSet>>
PointerObjectConstraintSet::Clone() const
{
  auto setClone = Set_.Clone();
  auto constraintClone = std::make_unique<PointerObjectConstraintSet>(*setClone);
  for (auto constraint : Constraints_)
    constraintClone->AddConstraint(constraint);
  return std::make_pair(std::move(setClone), std::move(constraintClone));
}

} // namespace jlm::llvm::aa

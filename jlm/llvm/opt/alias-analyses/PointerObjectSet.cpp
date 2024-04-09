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
PointerObjectSet::CanPointerObjectPoint(PointerObjectIndex index) const noexcept
{
  JLM_ASSERT(index < NumPointerObjects());
  return PointerObjects_[index].CanPoint();
}

bool
PointerObjectSet::CanPointerObjectBePointee(PointerObjectIndex index) const noexcept
{
  JLM_ASSERT(index < NumPointerObjects());
  return PointerObjects_[index].CanBePointee();
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
  JLM_ASSERT(index < NumPointerObjects());
  if (PointerObjects_[index].HasEscaped)
    return false;
  PointerObjects_[index].HasEscaped = true;

  // Pointer objects that have addresses can be written to from outside the module
  if (CanPointerObjectBePointee(index))
    MarkAsPointingToExternal(index);

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
  if (!CanPointerObjectPoint(index))
    return false;

  auto parent = GetUnificationRoot(index);
  if (PointerObjects_[parent].PointsToExternal)
    return false;
  PointerObjects_[parent].PointsToExternal = true;
  return true;
}

PointerObjectIndex
PointerObjectSet::GetUnificationRoot(PointerObjectIndex index) const noexcept
{
  if constexpr (ENABLE_UNIFICATION)
  {
    JLM_ASSERT(index < NumPointerObjects());

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

PointerObjectIndex
PointerObjectSet::UnifyPointerObjects(PointerObjectIndex object1, PointerObjectIndex object2)
{
  if constexpr (!ENABLE_UNIFICATION)
    JLM_UNREACHABLE("Unification is not enabled");

  JLM_ASSERT(CanPointerObjectPoint(object1));
  JLM_ASSERT(CanPointerObjectPoint(object2));

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

  if (IsPointingToExternal(oldRoot))
    MarkAsPointingToExternal(newRoot);

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
  // Assert the pointer object is a possible pointee
  JLM_ASSERT(CanPointerObjectBePointee(pointee));

  // If the pointer PointerObject can not point to anything, silently ignore
  if (!CanPointerObjectPoint(pointer))
    return false;

  return PointsToSets_[GetUnificationRoot(pointer)].Insert(pointee);
}

// Makes P(superset) a superset of P(subset)
bool
PointerObjectSet::MakePointsToSetSuperset(PointerObjectIndex superset, PointerObjectIndex subset)
{
  // If the superset PointerObject can't point to anything, silently ignore
  if (!CanPointerObjectPoint(superset))
    return false;

  // If the subset PointerObject can't point to anything, silently ignore
  if (!CanPointerObjectPoint(subset))
    return false;

  auto supersetParent = GetUnificationRoot(superset);
  auto subsetParent = GetUnificationRoot(subset);

  if (supersetParent == subsetParent)
    return false;

  auto & P_super = PointsToSets_[supersetParent];
  auto & P_sub = PointsToSets_[subsetParent];

  bool modified = P_super.UnionWith(P_sub);

  // If the external node is in the subset, it must also be part of the superset
  if (IsPointingToExternal(subsetParent))
    modified |= MarkAsPointingToExternal(supersetParent);

  return modified;
}

// Marks all x in P(pointer) as escaped
bool
PointerObjectSet::MarkAllPointeesAsEscaped(PointerObjectIndex pointer)
{
  bool modified = false;
  for (PointerObjectIndex pointee : GetPointsToSet(pointer).Items())
    modified |= MarkAsEscaped(pointee);

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

  // If external in P(Pointer1_), P(external) should become a superset of P(Pointer2)
  // In practice, this means everything in P(Pointer2) escapes
  if (set.IsPointingToExternal(Pointer_))
    modified |= set.MarkAllPointeesAsEscaped(Value_);

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
 * @param markAsEscaped the function to call when a PointerObject should be marked as escaped
 * @param markAsPointsToExternal called to flag a PointerObject as pointing to external
 */
template<typename MarkAsEscaped, typename MarkAsPointsToExternal>
void
HandleCallingExternalFunction(
    PointerObjectSet & set,
    const jlm::llvm::CallNode & callNode,
    MarkAsEscaped & markAsEscaped,
    MarkAsPointsToExternal & markAsPointsToExternal)
{

  // Mark all the call's inputs as escaped, and all the outputs as pointing to external
  for (size_t n = 0; n < callNode.NumArguments(); n++)
  {
    const auto & inputRegister = *callNode.Argument(n)->origin();
    const auto inputRegisterPO = set.TryGetRegisterPointerObject(inputRegister);

    if (inputRegisterPO)
      markAsEscaped(inputRegisterPO.value());
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
 * @param markAsEscaped the function to call when a PointerObject should be marked as escaped
 * @param markAsPointsToExternal called to flag a PointerObject as pointing to external
 */
template<typename MarkAsEscaped, typename MarkAsPointsToExternal>
static void
HandleCallingImportedFunction(
    PointerObjectSet & set,
    const jlm::llvm::CallNode & callNode,
    [[maybe_unused]] PointerObjectIndex imported,
    MarkAsEscaped & markAsEscaped,
    MarkAsPointsToExternal & markAsPointsToExternal)
{
  // FIXME: Add special handling of common library functions
  // Otherwise we don't know anything about the function
  return HandleCallingExternalFunction(set, callNode, markAsEscaped, markAsPointsToExternal);
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
template<typename MakeSuperset>
static void
HandleCallingLambdaFunction(
    PointerObjectSet & set,
    const jlm::llvm::CallNode & callNode,
    PointerObjectIndex lambda,
    MakeSuperset & makeSuperset)
{
  auto & lambdaNode = set.GetLambdaNodeFromFunctionMemoryObject(lambda);

  // If the number of parameters or number of results doesn't line up,
  // assume this is not the function we are calling.
  // Note that the number of arguments and results include 3 state edges: memory, loop and IO.
  // Varargs are properly handled, since they get merged by a valist_op node before the CallNode.
  if (lambdaNode.nfctarguments() != callNode.NumArguments()
      || lambdaNode.nfctresults() != callNode.NumResults())
    return;

  // Pass all call node inputs to the function's subregion
  for (size_t n = 0; n < callNode.NumArguments(); n++)
  {
    const auto & inputRegister = *callNode.Argument(n)->origin();
    const auto & argumentRegister = *lambdaNode.fctargument(n);

    const auto inputRegisterPO = set.TryGetRegisterPointerObject(inputRegister);
    const auto argumentRegisterPO = set.TryGetRegisterPointerObject(argumentRegister);
    if (!inputRegisterPO || !argumentRegisterPO)
      continue;

    makeSuperset(argumentRegisterPO.value(), inputRegisterPO.value());
  }

  // Pass the function's subregion results to the output of the call node
  for (size_t n = 0; n < callNode.NumResults(); n++)
  {
    const auto & outputRegister = *callNode.Result(n);
    const auto & resultRegister = *lambdaNode.fctresult(n)->origin();

    const auto outputRegisterPO = set.TryGetRegisterPointerObject(outputRegister);
    const auto resultRegisterPO = set.TryGetRegisterPointerObject(resultRegister);
    if (!outputRegisterPO || !resultRegisterPO)
      continue;

    makeSuperset(outputRegisterPO.value(), resultRegisterPO.value());
  }
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

  const auto MarkAsEscaped = [&](PointerObjectIndex index)
  {
    modified |= set.MarkAsEscaped(index);
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
      HandleCallingImportedFunction(set, CallNode_, target, MarkAsEscaped, MarkAsPointsToExternal);
    else if (kind == PointerObjectKind::FunctionMemoryObject)
      HandleCallingLambdaFunction(set, CallNode_, target, MakeSuperset);
  }

  // If we might be calling an external function
  if (set.IsPointingToExternal(Pointer_))
    HandleCallingExternalFunction(set, CallNode_, MarkAsEscaped, MarkAsPointsToExternal);

  return modified;
}

bool
EscapeFlagConstraint::PropagateEscapedFlagsDirectly(PointerObjectSet & set)
{
  std::queue<PointerObjectIndex> escapers;

  // First add all already escaped PointerObjects to the queue
  for (PointerObjectIndex idx = 0; idx < set.NumPointerObjects(); idx++)
  {
    if (set.HasEscaped(idx))
      escapers.push(idx);
  }

  bool modified = false;

  // For all escapers, check if they point to any PointerObjects not marked as escaped
  while (!escapers.empty())
  {
    const PointerObjectIndex escaper = escapers.front();
    escapers.pop();

    for (PointerObjectIndex pointee : set.GetPointsToSet(escaper).Items())
    {
      if (set.MarkAsEscaped(pointee))
      {
        // Add the newly marked PointerObject to the queue, in case the flag can be propagated
        escapers.push(pointee);
        modified = true;
      }
    }
  }

  return modified;
}

/**
 * Given an escaped function, the results should be marked as escaped,
 * and all arguments as pointing to external, provided they are of types we track the pointees of.
 * The modifications are made using the provided functors, which are called only if any flags are
 * missing. Each functor takes a single parameter:
 *   The index of a PointerObject of Register kind that is missing the specified flag.
 * @param set the PointerObjectSet representing this module
 * @param lambda the escaped PointerObject of function kind
 * @param markAsEscaped the function to call when a PointerObject should be marked as escaped
 * @param markAsPointsToExternal called to flag a PointerObject as pointing to external
 */
template<typename MarkAsEscapedFunctor, typename MarkAsPointsToExternalFunctor>
static void
HandleEscapedFunction(
    PointerObjectSet & set,
    PointerObjectIndex lambda,
    MarkAsEscapedFunctor & markAsEscaped,
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

    // Mark the result register as escaped
    markAsEscaped(resultPO.value());
  }
}

bool
EscapedFunctionConstraint::PropagateEscapedFunctionsDirectly(PointerObjectSet & set)
{
  bool modified = false;

  const auto markAsEscaped = [&](PointerObjectIndex index)
  {
    set.MarkAsEscaped(index);
    modified = true;
  };

  const auto markAsPointsToExternal = [&](PointerObjectIndex index)
  {
    set.MarkAsPointingToExternal(index);
    modified = true;
  };

  for (const auto [lambda, lambdaPO] : set.GetFunctionMap())
  {
    if (set.HasEscaped(lambdaPO))
      HandleEscapedFunction(set, lambdaPO, markAsEscaped, markAsPointsToExternal);
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
  // Registers themselves can't escape in the classical sense, since they don't have an address.
  // (CanBePointee() is false)
  // When marked as Escaped, it instead means that the contents of the register has escaped.
  // This allows Escaped-flag propagation to mark any pointee the register might hold as escaped.
  JLM_ASSERT(Set_.GetPointerObjectKind(registerIndex) == PointerObjectKind::Register);
  Set_.MarkAsEscaped(registerIndex);
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
  // If supersetEdges[x] contains y, (x -> y), that means P(y) supseteq P(x)
  std::vector<util::HashSet<PointerObjectIndex>> supersetEdges(Set_.NumPointerObjects());

  // Create quick lookup tables for Load, Store and function call constraints.
  // Lookup is indexed by the constraint's pointer
  std::vector<std::vector<PointerObjectIndex>> storeConstraints(Set_.NumPointerObjects());
  std::vector<std::vector<PointerObjectIndex>> loadConstraints(Set_.NumPointerObjects());
  std::vector<std::vector<const jlm::llvm::CallNode *>> callConstraints(Set_.NumPointerObjects());

  for (const auto & constraint : Constraints_)
  {
    if (const auto * ssConstraint = std::get_if<SupersetConstraint>(&constraint))
    {
      // Superset constraints become edges in the superset graph
      const auto superset = ssConstraint->GetSuperset();
      const auto subset = ssConstraint->GetSubset();
      JLM_ASSERT(superset < Set_.NumPointerObjects() && subset < Set_.NumPointerObjects());

      supersetEdges[subset].Insert(superset);
    }
    else if (const auto * storeConstraint = std::get_if<StoreConstraint>(&constraint))
    {
      const auto pointer = storeConstraint->GetPointer();
      const auto value = storeConstraint->GetValue();
      JLM_ASSERT(pointer < Set_.NumPointerObjects() && value < Set_.NumPointerObjects());

      storeConstraints[pointer].push_back(value);
    }
    else if (const auto * loadConstraint = std::get_if<LoadConstraint>(&constraint))
    {
      const auto pointer = loadConstraint->GetPointer();
      const auto value = loadConstraint->GetValue();
      JLM_ASSERT(pointer < Set_.NumPointerObjects() && value < Set_.NumPointerObjects());

      loadConstraints[pointer].push_back(value);
    }
    else if (const auto * callConstraint = std::get_if<FunctionCallConstraint>(&constraint))
    {
      const auto pointer = callConstraint->GetPointer();
      const auto & callNode = callConstraint->GetCallNode();
      JLM_ASSERT(pointer < Set_.NumPointerObjects());

      callConstraints[pointer].push_back(&callNode);
    }
  }

  // The worklist, initialized with every object
  util::LrfWorklist<PointerObjectIndex> worklist;
  for (PointerObjectIndex i = 0; i < Set_.NumPointerObjects(); i++)
    worklist.PushWorkItem(i);

  // Helper function for adding superset edges, propagating everything currently in the subset.
  // The superset is added to the work list if its points-to set or flags are changed.
  const auto AddSupersetEdge = [&](PointerObjectIndex superset, PointerObjectIndex subset)
  {
    // If the edge already exists, ignore
    if (!supersetEdges[subset].Insert(superset))
      return;

    // A new edge was added, propagate points to-sets
    if (!Set_.MakePointsToSetSuperset(superset, subset))
      return;

    // pointees or the points-to-external flag were propagated to the superset, add to the worklist
    worklist.PushWorkItem(superset);
  };

  // Helper function for flagging a pointer object as escaped. Adds to the worklist if changed
  const auto MarkAsEscaped = [&](PointerObjectIndex index)
  {
    if (Set_.MarkAsEscaped(index))
      worklist.PushWorkItem(index);
  };

  // Helper function for flagging a pointer as pointing to external. Adds to the worklist if changed
  const auto MarkAsPointsToExternal = [&](PointerObjectIndex index)
  {
    if (Set_.MarkAsPointingToExternal(index))
      worklist.PushWorkItem(index);
  };

  // Count of the total number of work items fired
  size_t numWorkItems = 0;

  // The main worklist loop. A work item can be in the worklist for the following reasons:
  // - It has never been fired
  // - It has pointees added since the last time it was fired
  // - It has been marked as pointing to external since last time it was fired
  // - It has been marked as escaping since the last time it was fired
  while (worklist.HasMoreWorkItems())
  {
    const auto n = worklist.PopWorkItem();
    numWorkItems++;

    // Stores on the form *n = value.
    for (const auto value : storeConstraints[n])
    {
      // This loop ensures *P(n) supseteq P(value)
      for (const auto pointee : Set_.GetPointsToSet(n).Items())
        AddSupersetEdge(pointee, value);

      // If P(n) contains "external", the contents of the written value escapes
      if (Set_.IsPointingToExternal(n))
        MarkAsEscaped(value);
    }

    // Loads on the form value = *n.
    for (const auto value : loadConstraints[n])
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
              MarkAsEscaped,
              MarkAsPointsToExternal);
        else if (kind == PointerObjectKind::FunctionMemoryObject)
          HandleCallingLambdaFunction(Set_, *callNode, pointee, AddSupersetEdge);
      }

      // If P(n) contains "external", handle calling external functions
      if (Set_.IsPointingToExternal(n))
        HandleCallingExternalFunction(Set_, *callNode, MarkAsEscaped, MarkAsPointsToExternal);
    }

    // Propagate P(n) along all edges n -> superset
    for (const auto superset : supersetEdges[n].Items())
    {
      if (Set_.MakePointsToSetSuperset(superset, n))
        worklist.PushWorkItem(superset);
    }

    // If escaped, propagate escaped flag to all pointees
    if (Set_.HasEscaped(n))
    {
      for (const auto pointee : Set_.GetPointsToSet(n).Items())
        MarkAsEscaped(pointee);

      // Escaped functions also need to flag arguments and results in the function body
      if (Set_.GetPointerObjectKind(n) == PointerObjectKind::FunctionMemoryObject)
        HandleEscapedFunction(Set_, n, MarkAsEscaped, MarkAsPointsToExternal);
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

/*
 * Copyright 2023, 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>

#include <jlm/llvm/opt/alias-analyses/DifferencePropagation.hpp>
#include <jlm/llvm/opt/alias-analyses/LazyCycleDetection.hpp>
#include <jlm/llvm/opt/alias-analyses/OnlineCycleDetection.hpp>
#include <jlm/util/Worklist.hpp>

#include <limits>
#include <queue>
#include <variant>

namespace jlm::llvm::aa
{

/**
 * Flag that enables unification logic.
 * When enabled, each points-to set lookup needs to perform a find operation.
 * When disabled, attempting to call UnifyPointerObjects panics.
 */
static constexpr bool ENABLE_UNIFICATION = true;

PointerObjectIndex
PointerObjectSet::AddPointerObject(PointerObjectKind kind, bool canPoint)
{
  JLM_ASSERT(PointerObjects_.size() < std::numeric_limits<PointerObjectIndex>::max());
  PointerObjectIndex index = PointerObjects_.size();

  PointerObjects_.emplace_back(kind, canPoint);
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

size_t
PointerObjectSet::NumRegisterPointerObjects() const noexcept
{
  return NumPointerObjectsOfKind(PointerObjectKind::Register);
}

size_t
PointerObjectSet::NumMemoryPointerObjects() const noexcept
{
  return NumPointerObjects() - NumRegisterPointerObjects();
}

size_t
PointerObjectSet::NumMemoryPointerObjectsCanPoint() const noexcept
{
  size_t count = 0;
  for (auto & pointerObject : PointerObjects_)
  {
    count += !pointerObject.IsRegister() && pointerObject.CanPoint();
  }
  return count;
}

PointerObjectIndex
PointerObjectSet::CreateRegisterPointerObject(const rvsdg::output & rvsdgOutput)
{
  JLM_ASSERT(RegisterMap_.count(&rvsdgOutput) == 0);
  return RegisterMap_[&rvsdgOutput] = AddPointerObject(PointerObjectKind::Register, true);
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
  return AddPointerObject(PointerObjectKind::Register, true);
}

PointerObjectIndex
PointerObjectSet::CreateAllocaMemoryObject(const rvsdg::Node & allocaNode, bool canPoint)
{
  JLM_ASSERT(AllocaMap_.count(&allocaNode) == 0);
  return AllocaMap_[&allocaNode] =
             AddPointerObject(PointerObjectKind::AllocaMemoryObject, canPoint);
}

PointerObjectIndex
PointerObjectSet::CreateMallocMemoryObject(const rvsdg::Node & mallocNode, bool canPoint)
{
  JLM_ASSERT(MallocMap_.count(&mallocNode) == 0);
  return MallocMap_[&mallocNode] =
             AddPointerObject(PointerObjectKind::MallocMemoryObject, canPoint);
}

PointerObjectIndex
PointerObjectSet::CreateGlobalMemoryObject(const delta::node & deltaNode, bool canPoint)
{
  JLM_ASSERT(GlobalMap_.count(&deltaNode) == 0);
  return GlobalMap_[&deltaNode] = AddPointerObject(PointerObjectKind::GlobalMemoryObject, canPoint);
}

PointerObjectIndex
PointerObjectSet::CreateFunctionMemoryObject(const rvsdg::LambdaNode & lambdaNode)
{
  JLM_ASSERT(!FunctionMap_.HasKey(&lambdaNode));
  const auto pointerObject = AddPointerObject(PointerObjectKind::FunctionMemoryObject, false);
  FunctionMap_.Insert(&lambdaNode, pointerObject);
  return pointerObject;
}

PointerObjectIndex
PointerObjectSet::GetFunctionMemoryObject(const rvsdg::LambdaNode & lambdaNode) const
{
  JLM_ASSERT(FunctionMap_.HasKey(&lambdaNode));
  return FunctionMap_.LookupKey(&lambdaNode);
}

const rvsdg::LambdaNode &
PointerObjectSet::GetLambdaNodeFromFunctionMemoryObject(PointerObjectIndex index) const
{
  JLM_ASSERT(FunctionMap_.HasValue(index));
  return *FunctionMap_.LookupValue(index);
}

PointerObjectIndex
PointerObjectSet::CreateImportMemoryObject(const GraphImport & importNode)
{
  JLM_ASSERT(ImportMap_.count(&importNode) == 0);

  // All import memory objects are marked as CanPoint() == false, as the analysis has no chance at
  // tracking the points-to set of pointers located in separate modules
  auto importMemoryObject = AddPointerObject(PointerObjectKind::ImportMemoryObject, false);
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

const std::unordered_map<const rvsdg::Node *, PointerObjectIndex> &
PointerObjectSet::GetAllocaMap() const noexcept
{
  return AllocaMap_;
}

const std::unordered_map<const rvsdg::Node *, PointerObjectIndex> &
PointerObjectSet::GetMallocMap() const noexcept
{
  return MallocMap_;
}

const std::unordered_map<const delta::node *, PointerObjectIndex> &
PointerObjectSet::GetGlobalMap() const noexcept
{
  return GlobalMap_;
}

const util::BijectiveMap<const rvsdg::LambdaNode *, PointerObjectIndex> &
PointerObjectSet::GetFunctionMap() const noexcept
{
  return FunctionMap_;
}

const std::unordered_map<const GraphImport *, PointerObjectIndex> &
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
PointerObjectSet::CanPoint(PointerObjectIndex index) const noexcept
{
  JLM_ASSERT(index < NumPointerObjects());
  return PointerObjects_[index].CanPoint();
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

bool
PointerObjectSet::CanTrackPointeesImplicitly(PointerObjectIndex index) const noexcept
{
  auto root = GetUnificationRoot(index);
  return PointerObjects_[root].CanTrackPointeesImplicitly();
}

bool
PointerObjectSet::MarkAsStoringAsScalar(PointerObjectIndex index)
{
  auto root = GetUnificationRoot(index);
  if (PointerObjects_[root].StoredAsScalar)
    return false;
  PointerObjects_[root].StoredAsScalar = true;
  return true;
}

[[nodiscard]] bool
PointerObjectSet::IsStoredAsScalar(PointerObjectIndex index) const noexcept
{
  return PointerObjects_[GetUnificationRoot(index)].StoredAsScalar;
}

bool
PointerObjectSet::MarkAsLoadingAsScalar(PointerObjectIndex index)
{
  auto root = GetUnificationRoot(index);
  if (PointerObjects_[root].LoadedAsScalar)
    return false;
  PointerObjects_[root].LoadedAsScalar = true;
  return true;
}

[[nodiscard]] bool
PointerObjectSet::IsLoadedAsScalar(PointerObjectIndex index) const noexcept
{
  return PointerObjects_[GetUnificationRoot(index)].LoadedAsScalar;
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

  // Ensure any flags set on the points-to set continue to be set in the new unification
  if (IsPointingToExternal(oldRoot))
    MarkAsPointingToExternal(newRoot);
  if (HasPointeesEscaping(oldRoot))
    MarkAsPointeesEscaping(newRoot);
  if (IsStoredAsScalar(oldRoot))
    MarkAsStoringAsScalar(newRoot);
  if (IsLoadedAsScalar(oldRoot))
    MarkAsLoadingAsScalar(newRoot);

  // Perform the actual unification
  PointerObjectParents_[oldRoot] = newRoot;

  // Copy over all pointees, and clean the pointee set from the old root
  auto & oldRootPointees = PointsToSets_[oldRoot];

  NumSetInsertionAttempts_ += oldRootPointees.Size();
  NumExplicitPointeesRemoved_ += oldRootPointees.Size();

  PointsToSets_[newRoot].UnionWithAndClear(oldRootPointees);

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

  NumSetInsertionAttempts_++;
  return PointsToSets_[pointerRoot].Insert(pointee);
}

// Makes P(superset) a superset of P(subset)
template<typename NewPointeeFunctor>
bool
PointerObjectSet::PropagateNewPointees(
    PointerObjectIndex superset,
    PointerObjectIndex subset,
    NewPointeeFunctor & onNewPointee)
{
  auto supersetRoot = GetUnificationRoot(superset);
  auto subsetRoot = GetUnificationRoot(subset);

  if (supersetRoot == subsetRoot)
    return false;

  auto & P_super = PointsToSets_[supersetRoot];
  auto & P_sub = PointsToSets_[subsetRoot];

  NumSetInsertionAttempts_ += P_sub.Size();

  bool modified = false;
  for (PointerObjectIndex pointee : P_sub.Items())
  {
    if (P_super.Insert(pointee))
    {
      onNewPointee(pointee);
      modified = true;
    }
  }

  // If the external node is in the subset, it must also be part of the superset
  if (IsPointingToExternal(subsetRoot))
    modified |= MarkAsPointingToExternal(supersetRoot);

  return modified;
}

bool
PointerObjectSet::MakePointsToSetSuperset(PointerObjectIndex superset, PointerObjectIndex subset)
{
  // NewPointee is a no-op
  const auto & NewPointee = [](PointerObjectIndex)
  {
  };
  return PropagateNewPointees(superset, subset, NewPointee);
}

bool
PointerObjectSet::MakePointsToSetSuperset(
    PointerObjectIndex superset,
    PointerObjectIndex subset,
    util::HashSet<PointerObjectIndex> & newPointees)
{
  const auto & NewPointee = [&](PointerObjectIndex pointee)
  {
    newPointees.Insert(pointee);
  };
  return PropagateNewPointees(superset, subset, NewPointee);
}

void
PointerObjectSet::RemoveAllPointees(PointerObjectIndex index)
{
  auto root = GetUnificationRoot(index);
  NumExplicitPointeesRemoved_ += PointsToSets_[root].Size();
  PointsToSets_[root].Clear();
}

bool
PointerObjectSet::IsPointingTo(PointerObjectIndex pointer, PointerObjectIndex pointee) const
{
  // Check if it is an implicit pointee
  if (IsPointingToExternal(pointer) && HasEscaped(pointee))
  {
    return true;
  }

  // Otherwise, check if it is an explicit pointee
  if (GetPointsToSet(pointer).Contains(pointee))
  {
    return true;
  }

  return false;
}

std::unique_ptr<PointerObjectSet>
PointerObjectSet::Clone() const
{
  return std::make_unique<PointerObjectSet>(*this);
}

bool
PointerObjectSet::HasIdenticalSolAs(const PointerObjectSet & other) const
{
  if (NumPointerObjects() != other.NumPointerObjects())
    return false;

  // Check that each pointer object has the same Sol set in both sets
  for (PointerObjectIndex i = 0; i < NumPointerObjects(); i++)
  {
    // Either i escapes in both sets, or in neither set
    if (HasEscaped(i) != other.HasEscaped(i))
      return false;

    // Either i points to external in both sets, or in neither set
    if (IsPointingToExternal(i) != other.IsPointingToExternal(i))
      return false;

    // Each explicit pointee of i in one set, should also be a pointee of i in the opposite set
    for (auto thisPointee : GetPointsToSet(i).Items())
    {
      if (!other.IsPointingTo(i, thisPointee))
        return false;
    }
    for (auto otherPointee : other.GetPointsToSet(i).Items())
    {
      if (!IsPointingTo(i, otherPointee))
        return false;
    }
  }
  return true;
}

size_t
PointerObjectSet::GetNumSetInsertionAttempts() const noexcept
{
  return NumSetInsertionAttempts_;
}

size_t
PointerObjectSet::GetNumExplicitPointeesRemoved() const noexcept
{
  return NumExplicitPointeesRemoved_;
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
  for (size_t n = 0; n < CallOperation::NumArguments(callNode); n++)
  {
    const auto & inputRegister = *callNode.Argument(n)->origin();
    const auto inputRegisterPO = set.TryGetRegisterPointerObject(inputRegister);

    if (inputRegisterPO)
      markAsPointeesEscaping(inputRegisterPO.value());
  }

  for (size_t n = 0; n < callNode.noutputs(); n++)
  {
    const auto & outputRegister = *callNode.output(n);
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
    const rvsdg::LambdaNode & lambdaNode,
    MakeSupersetFunctor & makeSuperset)
{
  auto lambdaArgs = lambdaNode.GetFunctionArguments();
  for (size_t n = 0; n < CallOperation::NumArguments(callNode) && n < lambdaArgs.size(); n++)
  {
    const auto & inputRegister = *callNode.Argument(n)->origin();
    const auto & argumentRegister = *lambdaArgs[n];

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
    const rvsdg::LambdaNode & lambdaNode,
    MakeSupersetFunctor & makeSuperset)
{
  auto lambdaResults = lambdaNode.GetFunctionResults();
  for (size_t n = 0; n < callNode.noutputs() && n < lambdaResults.size(); n++)
  {
    const auto & outputRegister = *callNode.output(n);
    const auto & resultRegister = *lambdaResults[n]->origin();

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
  bool modified = false;

  // First handle all unification roots marked as storing or loading scalars
  for (PointerObjectIndex idx = 0; idx < set.NumPointerObjects(); idx++)
  {
    if (!set.IsUnificationRoot(idx))
      continue;

    if (set.IsStoredAsScalar(idx))
    {
      for (auto pointee : set.GetPointsToSet(idx).Items())
      {
        modified |= set.MarkAsPointingToExternal(pointee);
      }
    }
    if (set.IsLoadedAsScalar(idx))
    {
      for (auto pointee : set.GetPointsToSet(idx).Items())
      {
        modified |= set.MarkAsPointeesEscaping(pointee);
      }
    }
  }

  std::queue<PointerObjectIndex> pointeeEscapers;
  // Add all unification roots marked as PointeesEscaping to the queue
  for (PointerObjectIndex idx = 0; idx < set.NumPointerObjects(); idx++)
  {
    if (set.IsUnificationRoot(idx) && set.HasPointeesEscaping(idx))
      pointeeEscapers.push(idx);
  }

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
  for (auto argument : lambdaNode.GetFunctionArguments())
  {
    // Argument registers that are mapped to a register pointer object should point to external
    const auto argumentPO = set.TryGetRegisterPointerObject(*argument);
    if (!argumentPO)
      continue;

    // Nothing to be done if it is already marked as points to external
    if (set.IsPointingToExternal(argumentPO.value()))
      continue;

    markAsPointsToExternal(argumentPO.value());
  }

  // All results of pointer type need to be flagged as pointees escaping
  for (auto result : lambdaNode.GetFunctionResults())
  {
    const auto resultPO = set.TryGetRegisterPointerObject(*result->origin());
    if (!resultPO)
      continue;

    // Nothing to be done if it is already marked as pointees escaping
    if (set.HasPointeesEscaping(resultPO.value()))
      continue;

    // Mark the result register as making any pointees it may have escape
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

  for (const auto & [lambda, lambdaPO] : set.GetFunctionMap())
  {
    if (set.HasEscaped(lambdaPO))
      HandleEscapedFunction(set, lambdaPO, markAsPointeesEscaping, markAsPointsToExternal);
  }

  return modified;
}

bool
PointerObjectConstraintSet::IsFrozen() const noexcept
{
  return ConstraintSetFrozen_;
}

void
PointerObjectConstraintSet::AddPointerPointeeConstraint(
    PointerObjectIndex pointer,
    PointerObjectIndex pointee)
{
  JLM_ASSERT(!IsFrozen());
  // All set constraints are additive, so simple constraints like this can be directly applied.
  Set_.AddToPointsToSet(pointer, pointee);
}

void
PointerObjectConstraintSet::AddPointsToExternalConstraint(PointerObjectIndex pointer)
{
  JLM_ASSERT(!IsFrozen());
  // Flags are never removed, so adding the flag now ensures it will be included.
  Set_.MarkAsPointingToExternal(pointer);
}

void
PointerObjectConstraintSet::AddRegisterContentEscapedConstraint(PointerObjectIndex registerIndex)
{
  JLM_ASSERT(!IsFrozen());
  JLM_ASSERT(Set_.IsPointerObjectRegister(registerIndex));
  Set_.MarkAsPointeesEscaping(registerIndex);
}

void
PointerObjectConstraintSet::AddConstraint(ConstraintVariant c)
{
  JLM_ASSERT(!IsFrozen());
  Constraints_.push_back(c);
}

const std::vector<PointerObjectConstraintSet::ConstraintVariant> &
PointerObjectConstraintSet::GetConstraints() const noexcept
{
  return Constraints_;
}

size_t
PointerObjectConstraintSet::NumBaseConstraints() const noexcept
{
  size_t numBaseConstraints = 0;
  for (PointerObjectIndex i = 0; i < Set_.NumPointerObjects(); i++)
  {
    if (Set_.IsUnificationRoot(i))
      numBaseConstraints += Set_.GetPointsToSet(i).Size();
  }
  return numBaseConstraints;
}

std::pair<size_t, size_t>
PointerObjectConstraintSet::NumFlagConstraints() const noexcept
{
  size_t numScalarFlagConstraints = 0;
  size_t numOtherFlagConstraints = 0;
  for (PointerObjectIndex i = 0; i < Set_.NumPointerObjects(); i++)
  {
    if (Set_.HasEscaped(i))
      numOtherFlagConstraints++;

    if (!Set_.IsUnificationRoot(i))
      continue;

    if (Set_.IsPointingToExternal(i))
      numOtherFlagConstraints++;
    if (Set_.HasPointeesEscaping(i))
      numOtherFlagConstraints++;
    if (Set_.IsStoredAsScalar(i))
      numScalarFlagConstraints++;
    if (Set_.IsLoadedAsScalar(i))
      numScalarFlagConstraints++;
  }
  return { numScalarFlagConstraints, numOtherFlagConstraints };
}

/**
 * Creates a label describing the PointerObject with the given \p index in the given \p set.
 * The label includes the index and the PointerObjectKind.
 * The PointerObject's pointees are included, or a reference to the unification root.
 * Helper function used by DrawSubsetGraph.
 */
static std::string
CreateSubsetGraphNodeLabel(PointerObjectSet & set, PointerObjectIndex index)
{
  std::ostringstream label;
  label << index;

  auto kind = set.GetPointerObjectKind(index);
  if (kind == PointerObjectKind::AllocaMemoryObject)
    label << " A";
  else if (kind == PointerObjectKind::MallocMemoryObject)
    label << " M";
  else if (kind == PointerObjectKind::FunctionMemoryObject)
    label << " F";
  else if (kind == PointerObjectKind::GlobalMemoryObject)
    label << " G";
  else if (kind == PointerObjectKind::ImportMemoryObject)
    label << " I";
  else if (kind != PointerObjectKind::Register)
    JLM_UNREACHABLE("Unknown PointerObject kind");

  label << "\n";

  if (set.IsUnificationRoot(index))
  {
    label << "{";
    bool sep = false;
    for (auto pointee : set.GetPointsToSet(index).Items())
    {
      if (sep)
        label << ", ";
      sep = true;
      label << pointee;
    }
    // Add a + if pointing to external
    if (set.IsPointingToExternal(index))
      label << (sep ? ", +" : "+");
    label << "}";

    if (set.HasPointeesEscaping(index))
      label << "e";
  }
  else
  {
    label << "#" << set.GetUnificationRoot(index);
  }

  if (!set.CanPoint(index))
    label << "\nCantPoint";

  return label.str();
}

/**
 * Creates GraphWriter nodes for each PointerObject in the set, with appropriate shape and label.
 * Memory objects are rectangular, registers are oval. Escaped nodes have a yellow fill.
 * Helper function used by DrawSubsetGraph.
 */
static void
CreateSubsetGraphNodes(PointerObjectSet & set, util::Graph & graph)
{
  // Ensure the index of nodes line up with the index of the corresponding PointerObject
  JLM_ASSERT(graph.NumNodes() == 0);

  // Create nodes for each PointerObject
  for (PointerObjectIndex i = 0; i < set.NumPointerObjects(); i++)
  {
    auto & node = graph.CreateNode();
    node.SetLabel(CreateSubsetGraphNodeLabel(set, i));

    if (set.IsPointerObjectRegister(i))
      node.SetShape(util::Node::Shape::Oval);
    else
      node.SetShape(util::Node::Shape::Rectangle);

    if (set.HasEscaped(i))
      node.SetFillColor("#FFFF99");
  }

  // Associate PointerObjects nodes with their associated RVSDG nodes / outputs
  for (auto [allocaNode, index] : set.GetAllocaMap())
    graph.GetNode(index).SetAttributeObject("rvsdgAlloca", *allocaNode);

  for (auto [mallocNode, index] : set.GetMallocMap())
    graph.GetNode(index).SetAttributeObject("rvsdgMalloc", *mallocNode);

  for (auto [deltaNode, index] : set.GetGlobalMap())
    graph.GetNode(index).SetAttributeObject("rvsdgDelta", *deltaNode);

  for (auto [lambdaNode, index] : set.GetFunctionMap())
    graph.GetNode(index).SetAttributeObject("rvsdgLambda", *lambdaNode);

  for (auto [importArgument, index] : set.GetImportMap())
    graph.GetNode(index).SetAttributeObject("rvsdgImport", *importArgument);

  // Multiple registers can be associated with the same register PointerObject, so add a suffix
  std::unordered_map<PointerObjectIndex, size_t> associationSuffix;
  for (auto [rvsdgOutput, index] : set.GetRegisterMap())
  {
    auto suffix = associationSuffix[index]++;
    graph.GetNode(index).SetAttributeObject(util::strfmt("rvsdgOutput", suffix), *rvsdgOutput);
  }
}

/**
 * Creates edges representing the different constraints in the subset graph.
 * Subset constraints become normal edges.
 * Load and store constraints become dashed edges with a circle on the end being dereferenced.
 * Function call constraints append to the labels to the nodes involved in the function call.
 * Helper function used by DrawSubsetGraph.
 */
static void
CreateSubsetGraphEdges(
    const PointerObjectSet & set,
    const std::vector<PointerObjectConstraintSet::ConstraintVariant> & constraints,
    util::Graph & graph)
{
  // Draw edges for constraints
  size_t nextCallConstraintIndex = 0;
  for (auto & constraint : constraints)
  {
    if (auto * supersetConstraint = std::get_if<SupersetConstraint>(&constraint))
    {
      graph.CreateDirectedEdge(
          graph.GetNode(supersetConstraint->GetSubset()),
          graph.GetNode(supersetConstraint->GetSuperset()));
    }
    else if (auto * storeConstraint = std::get_if<StoreConstraint>(&constraint))
    {
      auto & edge = graph.CreateDirectedEdge(
          graph.GetNode(storeConstraint->GetValue()),
          graph.GetNode(storeConstraint->GetPointer()));
      edge.SetStyle(util::Edge::Style::Dashed);
      edge.SetArrowHead("normalodot");
    }
    else if (auto * loadConstraint = std::get_if<LoadConstraint>(&constraint))
    {
      auto & edge = graph.CreateDirectedEdge(
          graph.GetNode(loadConstraint->GetPointer()),
          graph.GetNode(loadConstraint->GetValue()));
      edge.SetStyle(util::Edge::Style::Dashed);
      edge.SetArrowTail("odot");
    }
    else if (auto * callConstraint = std::get_if<FunctionCallConstraint>(&constraint))
    {
      auto callConstraintIndex = nextCallConstraintIndex++;
      auto & pointerNode = graph.GetNode(callConstraint->GetPointer());
      pointerNode.AppendToLabel(util::strfmt("call", callConstraintIndex, " target"));

      // Connect all registers that correspond to inputs and outputs of the call, to the call target
      auto & callNode = callConstraint->GetCallNode();
      for (size_t i = 0; i < CallOperation::NumArguments(callNode); i++)
      {
        if (auto inputRegister = set.TryGetRegisterPointerObject(*callNode.Argument(i)->origin()))
        {
          const auto label = util::strfmt("call", callConstraintIndex, " input", i);
          graph.GetNode(*inputRegister).AppendToLabel(label);
        }
      }
      for (size_t i = 0; i < callNode.noutputs(); i++)
      {
        if (auto outputRegister = set.TryGetRegisterPointerObject(*callNode.output(i)))
        {
          const auto label = util::strfmt("call", callConstraintIndex, " output", i);
          graph.GetNode(*outputRegister).AppendToLabel(label);
        }
      }
    }
    else
    {
      JLM_UNREACHABLE("Unknown constraint type");
    }
  }
}

/**
 * Appends to the labels of all nodes that represent parts of functions.
 * This includes nodes representing the function itself, e.g. "function4",
 * nodes representing the functions arguments, e.g. "function4 arg2",
 * and nodes representing values returned from the function, e.g. "function4 res0".
 * Helper function used by DrawSubsetGraph.
 */
static void
LabelFunctionsArgumentsAndReturnValues(PointerObjectSet & set, util::Graph & graph)
{
  size_t nextFunctionIndex = 0;
  for (auto [function, pointerObject] : set.GetFunctionMap())
  {
    JLM_ASSERT(set.GetPointerObjectKind(pointerObject) == PointerObjectKind::FunctionMemoryObject);
    const auto functionIndex = nextFunctionIndex++;
    graph.GetNode(pointerObject).AppendToLabel(util::strfmt("function", functionIndex));

    // Add labels to registers corresponding to arguments and results of the function
    auto args = function->GetFunctionArguments();
    for (size_t i = 0; i < args.size(); i++)
    {
      if (auto argumentRegister = set.TryGetRegisterPointerObject(*args[i]))
      {
        const auto label = util::strfmt("function", functionIndex, " arg", i);
        graph.GetNode(*argumentRegister).AppendToLabel(label);
      }
    }
    auto results = function->GetFunctionResults();
    for (size_t i = 0; i < results.size(); i++)
    {
      if (auto resultRegister = set.TryGetRegisterPointerObject(*results[i]->origin()))
      {
        const auto label = util::strfmt("function", functionIndex, " res", i);
        graph.GetNode(*resultRegister).AppendToLabel(label);
      }
    }
  }
}

util::Graph &
PointerObjectConstraintSet::DrawSubsetGraph(util::GraphWriter & writer) const
{
  auto & graph = writer.CreateGraph();
  graph.SetLabel("Andersen subset graph");

  CreateSubsetGraphNodes(Set_, graph);
  CreateSubsetGraphEdges(Set_, Constraints_, graph);
  LabelFunctionsArgumentsAndReturnValues(Set_, graph);

  return graph;
}

// Helper function for DoOfflineVariableSubstitution()
std::tuple<size_t, std::vector<util::HashSet<PointerObjectIndex>>, std::vector<bool>>
PointerObjectConstraintSet::CreateOvsSubsetGraph()
{
  // The index of n(v) is the index of the PointerObject v
  // The index of n(*v) is the index of v + derefNodeOffset
  const size_t derefNodeOffset = Set_.NumPointerObjects();
  const size_t totalNodeCount = Set_.NumPointerObjects() * 2;
  std::vector<util::HashSet<PointerObjectIndex>> successors(totalNodeCount);
  std::vector<bool> isDirectNode(totalNodeCount, false);

  // Nodes representing registers can be direct nodes, but only if they have empty points-to sets
  for (auto [_, index] : Set_.GetRegisterMap())
    isDirectNode[index] = Set_.GetPointsToSet(index).IsEmpty();

  // Mark all function argument register nodes as not direct
  for (auto [lambda, _] : Set_.GetFunctionMap())
  {
    for (auto arg : lambda->GetFunctionArguments())
    {
      if (auto argumentPO = Set_.TryGetRegisterPointerObject(*arg))
        isDirectNode[*argumentPO] = false;
    }
  }

  // Create the offline subset graph, and mark all registers that are not direct
  for (auto constraint : Constraints_)
  {
    if (auto * supersetConstraint = std::get_if<SupersetConstraint>(&constraint))
    {
      auto subset = Set_.GetUnificationRoot(supersetConstraint->GetSubset());
      auto superset = Set_.GetUnificationRoot(supersetConstraint->GetSuperset());

      successors[subset].Insert(superset);
      // Also add an edge for *subset -> *superset, from the original OVS paper.
      // It is not mentioned in Hardekopf and Lin, 2007: The Ant and the Grasshopper.
      successors[subset + derefNodeOffset].Insert(superset + derefNodeOffset);
    }
    else if (auto * storeConstraint = std::get_if<StoreConstraint>(&constraint))
    {
      auto pointer = Set_.GetUnificationRoot(storeConstraint->GetPointer());
      auto value = Set_.GetUnificationRoot(storeConstraint->GetValue());

      // Add an edge for value -> *pointer
      successors[value].Insert(pointer + derefNodeOffset);
    }
    else if (auto * loadConstraint = std::get_if<LoadConstraint>(&constraint))
    {
      auto value = Set_.GetUnificationRoot(loadConstraint->GetValue());
      auto pointer = Set_.GetUnificationRoot(loadConstraint->GetPointer());

      // Add an edge for *pointer -> value
      successors[pointer + derefNodeOffset].Insert(value);
    }
    else if (auto * callConstraint = std::get_if<FunctionCallConstraint>(&constraint))
    {
      auto & callNode = callConstraint->GetCallNode();
      // Mark all results of function calls as non-direct nodes
      for (size_t n = 0; n < callNode.noutputs(); n++)
      {
        if (auto resultPO = Set_.TryGetRegisterPointerObject(*callNode.output(n)))
          isDirectNode[*resultPO] = false;
      }
    }
    else
      JLM_UNREACHABLE("Unknown constraint variant");
  }

  return { totalNodeCount, std::move(successors), std::move(isDirectNode) };
}

/**
 * Part of Offline Variable Substitution, works on the OVS subset graph.
 * Takes the set of SCCs in the graph, and assigns equivalence set labels to each SCC.
 * If all predecessors of a direct SCC have the same equivalence set label, that label is used.
 * @return a vector of equivalence set labels assigned to each SCC
 * @see PointerObjectConstraintSet::IsOfflineVariableSubstitutionEnabled()
 */
static std::vector<int64_t>
AssignOvsEquivalenceSetLabels(
    std::vector<util::HashSet<PointerObjectIndex>> & successors,
    size_t numSccs,
    std::vector<size_t> & sccIndex,
    std::vector<size_t> & topologicalOrder,
    std::vector<bool> & sccHasDirectNodesOnly)
{
  // Visit all SCCs in topological order and assign equivalence set labels
  int64_t nextEquivalenceSetLabel = 0;
  const int64_t NO_EQUIVALENCE_SET_LABEL = -1;
  std::vector<int64_t> equivalenceSetLabels(numSccs, NO_EQUIVALENCE_SET_LABEL);

  // If all predecessors of a direct SCC share equivalence set label, use that label.
  const int64_t NO_PREDECESSOR_YET = -1;   // Value used when no predecessor label has been seen
  const int64_t SEVERAL_PREDECESSORS = -2; // Value used when different predecessors have been seen
  std::vector<int64_t> predecessorEquivalenceLabels(numSccs, NO_PREDECESSOR_YET);

  // Iterate over each SCC in topological order, and each node within the SCC.
  // This ensures all predecessor SCCs are known before visiting each SCC.
  for (auto node : topologicalOrder)
  {
    const auto scc = sccIndex[node];

    // If this SCC has not been visited in the topological order traversal, give it a label
    if (equivalenceSetLabels[scc] == NO_EQUIVALENCE_SET_LABEL)
    {
      // Check if the SCC is direct, and all predecessors share a single equivalence label.
      // Otherwise, give it a new unique equivalence set label.
      if (sccHasDirectNodesOnly[scc] && predecessorEquivalenceLabels[scc] != NO_PREDECESSOR_YET
          && predecessorEquivalenceLabels[scc] != SEVERAL_PREDECESSORS)
      {
        equivalenceSetLabels[scc] = predecessorEquivalenceLabels[scc];
      }
      else
      {
        equivalenceSetLabels[scc] = nextEquivalenceSetLabel++;
      }
    }

    // Inform all successors of this node about this SCC's equivalence label
    for (auto successor : successors[node].Items())
    {
      const auto successorSCC = sccIndex[successor];
      if (predecessorEquivalenceLabels[successorSCC] == SEVERAL_PREDECESSORS)
        continue;

      if (predecessorEquivalenceLabels[successorSCC] == NO_PREDECESSOR_YET)
      {
        predecessorEquivalenceLabels[successorSCC] = equivalenceSetLabels[scc];
      }
      else if (predecessorEquivalenceLabels[successorSCC] != equivalenceSetLabels[scc])
      {
        predecessorEquivalenceLabels[successorSCC] = SEVERAL_PREDECESSORS;
      }
    }
  }

  return equivalenceSetLabels;
}

size_t
PointerObjectConstraintSet::PerformOfflineVariableSubstitution(bool storeRefCycleUnificationRoot)
{
  // Performing unification on direct nodes relies on all subset edges being known offline.
  // This is only safe if no more constraints are added to the node in the future.
  ConstraintSetFrozen_ = true;

  // For each PointerObject v, creates two nodes: n(v) and n(*v), and creates edges between them
  auto subsetGraph = CreateOvsSubsetGraph();
  auto totalNodeCount = std::get<0>(subsetGraph);
  auto & successors = std::get<1>(subsetGraph);
  auto & isDirectNode = std::get<2>(subsetGraph);

  auto GetSuccessors = [&](PointerObjectIndex node)
  {
    return successors[node].Items();
  };

  // Output vectors from Tarjan's SCC algorithm
  std::vector<size_t> sccIndex;
  std::vector<size_t> topologicalOrder;
  auto numSccs = util::FindStronglyConnectedComponents(
      totalNodeCount,
      GetSuccessors,
      sccIndex,
      topologicalOrder);

  // Find out which SCCs contain only direct nodes, as described in CreateOvsSubsetGraph()
  std::vector<bool> sccHasDirectNodesOnly(numSccs, true);
  for (size_t node = 0; node < totalNodeCount; node++)
  {
    if (!isDirectNode[node])
      sccHasDirectNodesOnly[sccIndex[node]] = false;
  }

  // Give each SCC an equivalence set label
  auto equivalenceSetLabels = AssignOvsEquivalenceSetLabels(
      successors,
      numSccs,
      sccIndex,
      topologicalOrder,
      sccHasDirectNodesOnly);

  // Finally unify all PointerObjects with equal equivalence label
  size_t numUnifications = 0;
  std::vector<std::optional<PointerObjectIndex>> unificationRoot(
      equivalenceSetLabels.size(),
      std::nullopt);

  for (PointerObjectIndex i = 0; i < Set_.NumPointerObjects(); i++)
  {
    const auto equivalenceSetLabel = equivalenceSetLabels[sccIndex[i]];

    // If other nodes with the same equivalence set label have been seen, unify it with i
    if (unificationRoot[equivalenceSetLabel])
    {
      unificationRoot[equivalenceSetLabel] =
          Set_.UnifyPointerObjects(i, *unificationRoot[equivalenceSetLabel]);
      numUnifications++;
    }
    else
      unificationRoot[equivalenceSetLabel] = i;
  }

  // If hybrid cycle detection is enabled, it requires some information to be kept from OVS
  if (storeRefCycleUnificationRoot)
  {
    // For each ref node that is in a cycle with a regular node, store it for hybrid cycle detection
    // The idea: Any pointee of p should be unified with a, if *p and a are in the same SCC
    // NOTE: We do not use equivalence set labels here, as they represent more than just cycles

    // First find one unification root representing each SCC
    std::vector<std::optional<PointerObjectIndex>> unificationRootPerSCC(numSccs, std::nullopt);
    for (PointerObjectIndex i = 0; i < Set_.NumPointerObjects(); i++)
    {
      if (!unificationRootPerSCC[sccIndex[i]])
        unificationRootPerSCC[sccIndex[i]] = Set_.GetUnificationRoot(i);
    }

    // Assign unification roots to ref nodes that belong to SCCs with at least one regular node
    const size_t derefNodeOffset = Set_.NumPointerObjects();
    for (PointerObjectIndex i = 0; i < Set_.NumPointerObjects(); i++)
    {
      if (auto optRoot = unificationRootPerSCC[sccIndex[i + derefNodeOffset]])
      {
        RefNodeUnificationRoot_[i] = *optRoot;
      }
    }
  }

  return numUnifications;
}

size_t
PointerObjectConstraintSet::NormalizeConstraints()
{
  // The new list of constraints, preserving the order of constraints that are not deleted.
  std::vector<ConstraintVariant> newConstraints;

  // Sets used to avoid adding duplicates of any constraints
  util::HashSet<std::pair<PointerObjectIndex, PointerObjectIndex>> addedSupersetConstraints;
  util::HashSet<std::pair<PointerObjectIndex, PointerObjectIndex>> addedStoreConstraints;
  util::HashSet<std::pair<PointerObjectIndex, PointerObjectIndex>> addedLoadConstraints;
  util::HashSet<std::pair<PointerObjectIndex, const CallNode *>> addedCallConstraints;

  for (auto constraint : Constraints_)
  {
    // Update all PointerObjectIndex fields to point to unification roots
    if (auto * supersetConstraint = std::get_if<SupersetConstraint>(&constraint))
    {
      auto supersetRoot = Set_.GetUnificationRoot(supersetConstraint->GetSuperset());
      auto subsetRoot = Set_.GetUnificationRoot(supersetConstraint->GetSubset());

      // Skip no-op constraints
      if (supersetRoot == subsetRoot)
        continue;

      if (addedSupersetConstraints.Insert({ supersetRoot, subsetRoot }))
        newConstraints.emplace_back(SupersetConstraint(supersetRoot, subsetRoot));
    }
    else if (auto * storeConstraint = std::get_if<StoreConstraint>(&constraint))
    {
      auto pointerRoot = Set_.GetUnificationRoot(storeConstraint->GetPointer());
      auto valueRoot = Set_.GetUnificationRoot(storeConstraint->GetValue());

      if (addedStoreConstraints.Insert({ pointerRoot, valueRoot }))
        newConstraints.emplace_back(StoreConstraint(pointerRoot, valueRoot));
    }
    else if (auto * loadConstraint = std::get_if<LoadConstraint>(&constraint))
    {
      auto valueRoot = Set_.GetUnificationRoot(loadConstraint->GetValue());
      auto pointerRoot = Set_.GetUnificationRoot(loadConstraint->GetPointer());

      if (addedLoadConstraints.Insert({ pointerRoot, valueRoot }))
        newConstraints.emplace_back(LoadConstraint(valueRoot, pointerRoot));
    }
    else if (auto * functionCallConstraint = std::get_if<FunctionCallConstraint>(&constraint))
    {
      auto pointerRoot = Set_.GetUnificationRoot(functionCallConstraint->GetPointer());
      auto & callNode = functionCallConstraint->GetCallNode();

      if (addedCallConstraints.Insert({ pointerRoot, &callNode }))
        newConstraints.emplace_back(FunctionCallConstraint(pointerRoot, callNode));
    }
    else
      JLM_UNREACHABLE("Unknown Constraint variant");
  }

  size_t reduction = Constraints_.size() - newConstraints.size();
  Constraints_ = std::move(newConstraints);
  return reduction;
}

template<
    typename Worklist,
    bool EnableOnlineCycleDetection,
    bool EnableHybridCycleDetection,
    bool EnableLazyCycleDetection,
    bool EnableDifferencePropagation,
    bool EnablePreferImplicitPointees>
void
PointerObjectConstraintSet::RunWorklistSolver(WorklistStatistics & statistics)
{
  // Check that the provided worklist implementation inherits from Worklist
  static_assert(std::is_base_of_v<util::Worklist<PointerObjectIndex>, Worklist>);

  // Online cycle detections detects all cycles immediately, so there is no point in enabling others
  if constexpr (EnableOnlineCycleDetection)
  {
    static_assert(!EnableHybridCycleDetection, "OnlineCD can not be combined with HybridCD");
    static_assert(!EnableLazyCycleDetection, "OnlineCD can not be combined with LazyCD");
  }

  // Create auxiliary subset graph.
  // All edges must have their tail be a unification root (non-root nodes have no successors).
  // If supersetEdges[x] contains y, (x -> y), that means P(y) supseteq P(x)
  std::vector<util::HashSet<PointerObjectIndex>> supersetEdges(Set_.NumPointerObjects());

  // Create quick lookup tables for Load, Store and function call constraints.
  // Lookup is indexed by the constraint's pointer.
  // The constraints need to be added to the unification root, as only unification roots
  // are allowed on the worklist. The sets are empty for all non-root nodes.
  std::vector<util::HashSet<PointerObjectIndex>> storeConstraints(Set_.NumPointerObjects());
  std::vector<util::HashSet<PointerObjectIndex>> loadConstraints(Set_.NumPointerObjects());
  std::vector<util::HashSet<const jlm::llvm::CallNode *>> callConstraints(Set_.NumPointerObjects());

  for (const auto & constraint : Constraints_)
  {
    if (const auto * ssConstraint = std::get_if<SupersetConstraint>(&constraint))
    {
      // Superset constraints become edges in the subset graph
      // When initializing the set of superset edges, normalize them as well
      auto superset = Set_.GetUnificationRoot(ssConstraint->GetSuperset());
      auto subset = Set_.GetUnificationRoot(ssConstraint->GetSubset());

      if (superset != subset) // Ignore self-edges
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

      callConstraints[pointer].Insert(&callNode);
    }
  }

  DifferencePropagation differencePropagation(Set_);
  if constexpr (EnableDifferencePropagation)
    differencePropagation.Initialize();

  // Makes pointer point to pointee.
  // Returns true if the pointee was new. Does not add pointer to the worklist.
  const auto & AddToPointsToSet = [&](PointerObjectIndex pointer,
                                      PointerObjectIndex pointee) -> bool
  {
    if constexpr (EnableDifferencePropagation)
      return differencePropagation.AddToPointsToSet(pointer, pointee);
    else
      return Set_.AddToPointsToSet(pointer, pointee);
  };

  // Makes superset point to everything subset points to, and propagates the PointsToEscaped flag.
  // Returns true if any pointees were new, or the flag was new.
  // Does not add superset to the worklist.
  const auto & MakePointsToSetSuperset = [&](PointerObjectIndex superset,
                                             PointerObjectIndex subset) -> bool
  {
    if constexpr (EnableDifferencePropagation)
      return differencePropagation.MakePointsToSetSuperset(superset, subset);
    else
      return Set_.MakePointsToSetSuperset(superset, subset);
  };

  // Performs unification safely while the worklist algorithm is running.
  // Ensures all constraints end up being owned by the new root.
  // It does NOT redirect constraints owned by other nodes, referencing a or b.
  // If a and b already belong to the same unification root, this is a no-op.
  // This operation does not add the unification result to the worklist.
  // Returns the root of the new unification, or the existing root if a and b were already unified.
  const auto UnifyPointerObjects = [&](PointerObjectIndex a,
                                       PointerObjectIndex b) -> PointerObjectIndex
  {
    const auto aRoot = Set_.GetUnificationRoot(a);
    const auto bRoot = Set_.GetUnificationRoot(b);

    if (aRoot == bRoot)
      return aRoot;

    const auto root = Set_.UnifyPointerObjects(aRoot, bRoot);
    // The root among the two original roots that did NOT end up as the new root
    const auto nonRoot = root == aRoot ? bRoot : aRoot;

    // Move constraints owned by the non-root to the root
    supersetEdges[root].UnionWithAndClear(supersetEdges[nonRoot]);

    // Try to avoid self-edges, but indirect self-edges can still exist
    supersetEdges[root].Remove(root);
    supersetEdges[root].Remove(nonRoot);

    storeConstraints[root].UnionWithAndClear(storeConstraints[nonRoot]);

    loadConstraints[root].UnionWithAndClear(loadConstraints[nonRoot]);

    callConstraints[root].UnionWithAndClear(callConstraints[nonRoot]);

    if constexpr (EnableDifferencePropagation)
      differencePropagation.OnPointerObjectsUnified(root, nonRoot);

    if constexpr (EnableHybridCycleDetection)
    {
      // If the new root did not have a ref node unification target, check if the other node has one
      if (RefNodeUnificationRoot_.count(root) == 0)
      {
        const auto nonRootRefUnification = RefNodeUnificationRoot_.find(nonRoot);
        if (nonRootRefUnification != RefNodeUnificationRoot_.end())
          RefNodeUnificationRoot_[root] = nonRootRefUnification->second;
      }
    }

    return root;
  };

  // Removes all explicit pointees from the given PointerObject
  const auto RemoveAllPointees = [&](PointerObjectIndex index)
  {
    JLM_ASSERT(Set_.IsUnificationRoot(index));
    Set_.RemoveAllPointees(index);

    // Prevent the difference propagation from keeping any pointees after the removal
    if constexpr (EnableDifferencePropagation)
      differencePropagation.OnRemoveAllPointees(index);
  };

  // Lambda for getting all superset edge successors of a given pointer object in the subset graph.
  // If \p node is not a unification root, its set of successors will always be empty.
  const auto GetSupersetEdgeSuccessors = [&](PointerObjectIndex node)
  {
    return supersetEdges[node].Items();
  };

  // If online cycle detection is enabled, perform the initial topological sorting now,
  // which may detect and eliminate cycles
  OnlineCycleDetector onlineCycleDetector(Set_, GetSupersetEdgeSuccessors, UnifyPointerObjects);
  if constexpr (EnableOnlineCycleDetection)
    onlineCycleDetector.InitializeTopologicalOrdering();

  // If lazy cycle detection is enabled, initialize it here
  LazyCycleDetector lazyCycleDetector(Set_, GetSupersetEdgeSuccessors, UnifyPointerObjects);
  if constexpr (EnableLazyCycleDetection)
    lazyCycleDetector.Initialize();

  if constexpr (EnableHybridCycleDetection)
    statistics.NumHybridCycleUnifications = 0;

  if constexpr (EnablePreferImplicitPointees)
    statistics.NumPipExplicitPointeesRemoved = 0;

  // The worklist, initialized with every unification root
  Worklist worklist;
  for (PointerObjectIndex i = 0; i < Set_.NumPointerObjects(); i++)
  {
    if (Set_.IsUnificationRoot(i))
      worklist.PushWorkItem(i);
  }

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

  // Helper function for adding superset edges, propagating everything currently in the subset.
  // The superset's root is added to the worklist if its points-to set or flags are changed.
  const auto AddSupersetEdge = [&](PointerObjectIndex superset, PointerObjectIndex subset)
  {
    superset = Set_.GetUnificationRoot(superset);
    subset = Set_.GetUnificationRoot(subset);

    // If this is a self-edge, ignore
    if (superset == subset)
      return;

    if constexpr (EnablePreferImplicitPointees)
    {
      // No need to add edges when all pointees propagate implicitly either way
      if (Set_.IsPointingToExternal(superset) && Set_.HasPointeesEscaping(subset))
      {
        return;
      }

      // Ignore adding simple edges to nodes that should only have implicit pointees
      if (Set_.CanTrackPointeesImplicitly(superset))
      {
        MarkAsPointeesEscaping(subset);
        return;
      }
    }

    // If the edge already exists, ignore
    if (!supersetEdges[subset].Insert(superset))
      return;

    // The edge is now added. If OCD is enabled, check if it broke the topological order, and fix it
    if constexpr (EnableOnlineCycleDetection)
    {
      // If a cycle is detected, this function eliminates it by unifying, and returns the root
      auto optUnificationRoot = onlineCycleDetector.MaintainTopologicalOrder(subset, superset);
      if (optUnificationRoot)
      {
        worklist.PushWorkItem(*optUnificationRoot);
        return;
      }
    }

    // A new edge was added, propagate points to-sets. If the superset changes, add to the worklist
    bool anyPropagation = MakePointsToSetSuperset(superset, subset);
    if (anyPropagation)
      worklist.PushWorkItem(superset);

    // If nothing was propagated by adding the edge, try lazy cycle detection
    if (EnableLazyCycleDetection && !Set_.GetPointsToSet(subset).IsEmpty() && !anyPropagation)
    {
      const auto optUnificationRoot = lazyCycleDetector.OnPropagatedNothing(subset, superset);
      if (optUnificationRoot)
        worklist.PushWorkItem(*optUnificationRoot);
    }
  };

  // A temporary place to store new subset edges, to avoid modifying sets while they are iterated
  util::HashSet<std::pair<PointerObjectIndex, PointerObjectIndex>> newSupersetEdges;
  const auto QueueNewSupersetEdge = [&](PointerObjectIndex superset, PointerObjectIndex subset)
  {
    superset = Set_.GetUnificationRoot(superset);
    subset = Set_.GetUnificationRoot(subset);
    if (superset == subset || supersetEdges[subset].Contains(superset))
      return;
    newSupersetEdges.Insert({ superset, subset });
  };

  const auto FlushNewSupersetEdges = [&]()
  {
    for (auto [superset, subset] : newSupersetEdges.Items())
      AddSupersetEdge(superset, subset);
    newSupersetEdges.Clear();
  };

  // Ensure that all functions that have already escaped have informed their arguments and results
  // The worklist will only inform functions if their HasEscaped flag changes
  EscapedFunctionConstraint::PropagateEscapedFunctionsDirectly(Set_);

  // The main work item handler. A work item can be in the worklist for the following reasons:
  // - It has never been fired
  // - It has pointees added since the last time it was fired
  // - It has been marked as pointing to external since last time it was fired
  // - It has been marked as escaping all pointees since last time it was fired
  // - It is the unification root of a new unification
  // Work items should be unification roots. If a work item is not a root when popped, it is skipped
  const auto HandleWorkItem = [&](PointerObjectIndex node)
  {
    statistics.NumWorkItemsPopped++;

    // Skip visiting unification roots.
    // All unification operations are responsible for adding the new root to the worklist if needed
    if (!Set_.IsUnificationRoot(node))
      return;

    // If difference propagation is enabled, this set contains only pointees that have been added
    // since the last time this work item was popped. Otherwise, it contains all pointees.
    const auto & newPointees = EnableDifferencePropagation
                                 ? differencePropagation.GetNewPointees(node)
                                 : Set_.GetPointsToSet(node);
    statistics.NumWorkItemNewPointees += newPointees.Size();

    // If difference propagation is enabled, this bool is true if this is the first time node
    // is being visited by the worklist with the PointsToExternal flag set
    const auto newPointsToExternal = EnableDifferencePropagation
                                       ? differencePropagation.PointsToExternalIsNew(node)
                                       : Set_.IsPointingToExternal(node);

    // Perform hybrid cycle detection if all pointees of node should be unified
    if constexpr (EnableHybridCycleDetection)
    {
      // If all pointees of node should be unified, do it now
      const auto & refUnificationRootIt = RefNodeUnificationRoot_.find(node);
      if (refUnificationRootIt != RefNodeUnificationRoot_.end())
      {
        auto & refUnificationRoot = refUnificationRootIt->second;
        // The ref unification root may no longer be a root, so update it first
        refUnificationRoot = Set_.GetUnificationRoot(refUnificationRoot);

        // if any unification happens, the result must be added to the worklist
        bool anyUnification = false;

        // Make a copy of the set, as the node itself may be unified, invalidating newPointees
        auto unificationMembers = newPointees;
        for (const auto pointee : unificationMembers.Items())
        {
          const auto pointeeRoot = Set_.GetUnificationRoot(pointee);
          if (pointeeRoot == refUnificationRoot)
            continue;

          (*statistics.NumHybridCycleUnifications)++;
          anyUnification = true;
          refUnificationRoot = UnifyPointerObjects(refUnificationRoot, pointeeRoot);
        }

        if (anyUnification)
        {
          JLM_ASSERT(Set_.IsUnificationRoot(refUnificationRoot));
          worklist.PushWorkItem(refUnificationRoot);
          // If the node itself was unified, the new root has been added to the worklist, so exit
          if (refUnificationRoot == Set_.GetUnificationRoot(node))
            return;
        }
      }
    }

    // If propagating to any node with AllPointeesEscape, we should have AllPointeesEscape
    if (EnablePreferImplicitPointees && !Set_.HasPointeesEscaping(node))
    {
      for (auto superset : supersetEdges[node].Items())
      {
        if (Set_.HasPointeesEscaping(superset))
        {
          // Mark the current node.
          // This is the beginning of the work item visit, so node does not need to be added again
          Set_.MarkAsPointeesEscaping(node);
          break;
        }
      }
    }

    const auto pointeesEscaping = Set_.HasPointeesEscaping(node);
    // If difference propagation is enabled, this bool is true if this is the first time node
    // is being visited by the worklist with the PointeesEscaping flag set
    const auto newPointeesEscaping = EnableDifferencePropagation
                                       ? differencePropagation.PointeesEscapeIsNew(node)
                                       : pointeesEscaping;

    // Mark pointees as escaping, if node has the PointeesEscaping flag
    if (pointeesEscaping)
    {
      // If this is the first time node is being visited with the PointeesEscaping flag set,
      // add the escaped flag to all pointees. Otherwise, only add it to new pointees.
      const auto & newEscapingPointees =
          newPointeesEscaping ? Set_.GetPointsToSet(node) : newPointees;
      for (const auto pointee : newEscapingPointees.Items())
      {
        const auto pointeeRoot = Set_.GetUnificationRoot(pointee);

        // Marking a node as escaped will imply two flags on the unification root:
        // - PointeesEscaping
        // - PointsToExternal
        const bool rootAlreadyHasFlags =
            Set_.HasPointeesEscaping(pointeeRoot) && Set_.IsPointingToExternal(pointeeRoot);

        // Mark the pointee itself as escaped, not the pointee's unifiction root!
        if (!Set_.MarkAsEscaped(pointee))
          continue;

        // If the PointerObject we just marked as escaped is a function, inform it about escaping
        if (Set_.GetPointerObjectKind(pointee) == PointerObjectKind::FunctionMemoryObject)
          HandleEscapedFunction(Set_, pointee, MarkAsPointeesEscaping, MarkAsPointsToExternal);

        // If the pointee's unification root previously didn't have both the flags implied by
        // having one of the unification members escaping, add the root to the worklist
        if (!rootAlreadyHasFlags)
        {
          JLM_ASSERT(Set_.HasPointeesEscaping(pointeeRoot));
          JLM_ASSERT(Set_.IsPointingToExternal(pointeeRoot));
          worklist.PushWorkItem(pointeeRoot);
        }
      }
    }

    // If this node can track all pointees implicitly, remove its explicit nodes
    if (EnablePreferImplicitPointees && Set_.CanTrackPointeesImplicitly(node))
    {
      *(statistics.NumPipExplicitPointeesRemoved) += Set_.GetPointsToSet(node).Size();
      // This also causes newPointees to become empty
      RemoveAllPointees(node);
    }

    // Propagate P(n) along all edges n -> superset
    auto supersets = supersetEdges[node].Items();
    for (auto it = supersets.begin(); it != supersets.end();)
    {
      const auto supersetParent = Set_.GetUnificationRoot(*it);

      // Remove self-edges
      if (supersetParent == node)
      {
        it = supersetEdges[node].Erase(it);
        continue;
      }

      // Remove edges from nodes with "all pointees escape" to nodes with "points to all escaped"
      if (EnablePreferImplicitPointees && pointeesEscaping
          && Set_.IsPointingToExternal(supersetParent))
      {
        it = supersetEdges[node].Erase(it);
        continue;
      }

      // The current it-edge should be kept as is, prepare "it" for the next iteration.
      ++it;

      bool modified = false;
      for (const auto pointee : newPointees.Items())
        modified |= AddToPointsToSet(supersetParent, pointee);

      if (newPointsToExternal)
        modified |= Set_.MarkAsPointingToExternal(supersetParent);

      if (modified)
        worklist.PushWorkItem(supersetParent);

      if (EnableLazyCycleDetection && !newPointees.IsEmpty() && !modified)
      {
        // If nothing was propagated along this edge, check if there is a cycle
        // If a cycle is detected, this function eliminates it by unifying, and returns the root
        auto optUnificationRoot = lazyCycleDetector.OnPropagatedNothing(node, supersetParent);
        if (optUnificationRoot)
        {
          // The new unification root is pushed, and handling of the current work item is aborted.
          worklist.PushWorkItem(*optUnificationRoot);
          return;
        }
      }
    }

    // Stores on the form *n = value.
    for (const auto value : storeConstraints[node].Items())
    {
      // This loop ensures *P(n) supseteq P(value)
      for (const auto pointee : newPointees.Items())
        QueueNewSupersetEdge(pointee, value);

      // If P(n) contains "external", the contents of the written value escapes
      if (newPointsToExternal)
        MarkAsPointeesEscaping(value);
    }

    // If node has the stored as scalar constraint, but does not make its pointees escape outright
    if (Set_.IsStoredAsScalar(node) && !Set_.HasPointeesEscaping(node))
    {
      for (const auto pointee : newPointees.Items())
      {
        MarkAsPointsToExternal(pointee);
      }
    }

    // Loads on the form value = *n.
    for (const auto value : loadConstraints[node].Items())
    {
      // This loop ensures P(value) supseteq *P(n)
      for (const auto pointee : newPointees.Items())
        QueueNewSupersetEdge(value, pointee);

      // If P(n) contains "external", the loaded value may also point to external
      if (newPointsToExternal)
        MarkAsPointsToExternal(value);
    }

    // If node has the loaded as scalar constraint, but does not make its pointees escape outright
    if (Set_.IsLoadedAsScalar(node) && !Set_.HasPointeesEscaping(node))
    {
      for (const auto pointee : newPointees.Items())
      {
        MarkAsPointeesEscaping(pointee);
      }
    }

    // Function calls on the form (*n)()
    for (const auto callNode : callConstraints[node].Items())
    {
      // Connect the inputs and outputs of the callNode to every possible function pointee
      for (const auto pointee : newPointees.Items())
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
          HandleCallingLambdaFunction(Set_, *callNode, pointee, QueueNewSupersetEdge);
      }

      // If P(n) contains "external", handle calling external functions
      if (newPointsToExternal)
        HandleCallingExternalFunction(
            Set_,
            *callNode,
            MarkAsPointeesEscaping,
            MarkAsPointsToExternal);
    }

    // No pointees have been added to P(node) while visiting node thus far in the handler.
    // All new flags have also been handled, or caused this node to be on the worklist again.
    if constexpr (EnableDifferencePropagation)
    {
      differencePropagation.ClearNewPointees(node);
      if (newPointsToExternal)
        differencePropagation.MarkPointsToExternalAsHandled(node);
      if (newPointeesEscaping)
        differencePropagation.MarkPointeesEscapeAsHandled(node);
    }

    // Add all new superset edges, which also propagates points-to sets immediately
    // and possibly performs unifications to eliminate cycles.
    // Any unified nodes, or nodes with updated points-to sets, are added to the worklist.
    FlushNewSupersetEdges();
  };

  // The Workset worklist only remembers which work items have been pushed.
  // It does not provide an iteration order, so if any work item need to be revisited,
  // we do a topological traversal over all work items instead, visiting ones in the Workset.
  // Performing topological sorting also detects all cycles, which are unified away.
  constexpr bool useTopologicalTraversal =
      std::is_same_v<Worklist, util::Workset<PointerObjectIndex>>;

  if constexpr (useTopologicalTraversal)
  {
    std::vector<PointerObjectIndex> sccIndex;
    std::vector<PointerObjectIndex> topologicalOrder;

    statistics.NumTopologicalWorklistSweeps = 0;

    while (worklist.HasMoreWorkItems())
    {
      (*statistics.NumTopologicalWorklistSweeps)++;

      // First perform a topological sort of the entire subset graph, with respect to simple edges
      util::FindStronglyConnectedComponents<PointerObjectIndex>(
          Set_.NumPointerObjects(),
          GetSupersetEdgeSuccessors,
          sccIndex,
          topologicalOrder);

      // Visit nodes in topological order, if they are in the workset.
      // cycles will result in neighbouring nodes in the topologicalOrder sharing sccIndex
      for (size_t i = 0; i < topologicalOrder.size(); i++)
      {
        const auto node = topologicalOrder[i];
        const auto nextNodeIndex = i + 1;
        if (nextNodeIndex < topologicalOrder.size())
        {
          auto & nextNode = topologicalOrder[nextNodeIndex];
          if (sccIndex[node] == sccIndex[nextNode])
          {
            // This node is in a cycle with the next node, unify them
            nextNode = UnifyPointerObjects(node, nextNode);
            // Make sure the new root is visited
            worklist.RemoveWorkItem(node);
            worklist.PushWorkItem(nextNode);
            continue;
          }
        }

        // If this work item is in the workset, handle it. Repeat immediately if it gets re-added.
        while (worklist.HasWorkItem(node))
        {
          worklist.RemoveWorkItem(node);
          HandleWorkItem(node);
        }
      }
    }
  }
  else
  {
    // The worklist is a normal worklist
    while (worklist.HasMoreWorkItems())
      HandleWorkItem(worklist.PopWorkItem());
  }

  if constexpr (EnableOnlineCycleDetection)
  {
    statistics.NumOnlineCyclesDetected = onlineCycleDetector.NumOnlineCyclesDetected();
    statistics.NumOnlineCycleUnifications = onlineCycleDetector.NumOnlineCycleUnifications();
  }

  if constexpr (EnableLazyCycleDetection)
  {
    statistics.NumLazyCyclesDetectionAttempts = lazyCycleDetector.NumCycleDetectionAttempts();
    statistics.NumLazyCyclesDetected = lazyCycleDetector.NumCyclesDetected();
    statistics.NumLazyCycleUnifications = lazyCycleDetector.NumCycleUnifications();
  }
}

PointerObjectConstraintSet::WorklistStatistics
PointerObjectConstraintSet::SolveUsingWorklist(
    WorklistSolverPolicy policy,
    bool enableOnlineCycleDetection,
    bool enableHybridCycleDetection,
    bool enableLazyCycleDetection,
    bool enableDifferencePropagation,
    bool enablePreferImplicitPointees)
{

  // Takes all parameters as compile time types.
  // tWorklist is a pointer to one of the Worklist implementations.
  // the rest are instances of std::bool_constant, either std::true_type or std::false_type
  const auto Dispatch = [&](auto tWorklist,
                            auto tOnlineCycleDetection,
                            auto tHybridCycleDetection,
                            auto tLazyCycleDetection,
                            auto tDifferencePropagation,
                            auto tPreferImplicitPointees) -> WorklistStatistics
  {
    using Worklist = std::remove_pointer_t<decltype(tWorklist)>;
    constexpr bool vOnlineCycleDetection = decltype(tOnlineCycleDetection)::value;
    constexpr bool vHybridCycleDetection = decltype(tHybridCycleDetection)::value;
    constexpr bool vLazyCycleDetection = decltype(tLazyCycleDetection)::value;
    constexpr bool vDifferencePropagation = decltype(tDifferencePropagation)::value;
    constexpr bool vPreferImplicitPointees = decltype(tPreferImplicitPointees)::value;

    if constexpr (
        std::is_same_v<Worklist, util::Workset<PointerObjectIndex>>
        && (vOnlineCycleDetection || vHybridCycleDetection || vLazyCycleDetection))
    {
      JLM_UNREACHABLE("Can not enable online, hybrid or lazy cycle detection with the topo policy");
    }
    if constexpr (vOnlineCycleDetection && (vHybridCycleDetection || vLazyCycleDetection))
    {
      JLM_UNREACHABLE("Can not enable hybrid or lazy cycle detection with online cycle detection");
    }
    else
    {
      WorklistStatistics statistics(policy);
      RunWorklistSolver<
          Worklist,
          vOnlineCycleDetection,
          vHybridCycleDetection,
          vLazyCycleDetection,
          vDifferencePropagation,
          vPreferImplicitPointees>(statistics);
      return statistics;
    }
  };

  std::variant<
      util::LrfWorklist<PointerObjectIndex> *,
      util::TwoPhaseLrfWorklist<PointerObjectIndex> *,
      util::Workset<PointerObjectIndex> *,
      util::LifoWorklist<PointerObjectIndex> *,
      util::FifoWorklist<PointerObjectIndex> *>
      policyVariant;

  if (policy == WorklistSolverPolicy::LeastRecentlyFired)
    policyVariant = (util::LrfWorklist<PointerObjectIndex> *)nullptr;
  else if (policy == WorklistSolverPolicy::TwoPhaseLeastRecentlyFired)
    policyVariant = (util::TwoPhaseLrfWorklist<PointerObjectIndex> *)nullptr;
  else if (policy == WorklistSolverPolicy::TopologicalSort)
    policyVariant = (util::Workset<PointerObjectIndex> *)nullptr;
  else if (policy == WorklistSolverPolicy::LastInFirstOut)
    policyVariant = (util::LifoWorklist<PointerObjectIndex> *)nullptr;
  else if (policy == WorklistSolverPolicy::FirstInFirstOut)
    policyVariant = (util::FifoWorklist<PointerObjectIndex> *)nullptr;
  else
    JLM_UNREACHABLE("Unknown worklist policy");

  std::variant<std::true_type, std::false_type> onlineCycleDetectionVariant;
  if (enableOnlineCycleDetection)
    onlineCycleDetectionVariant = std::true_type{};
  else
    onlineCycleDetectionVariant = std::false_type{};

  std::variant<std::true_type, std::false_type> hybridCycleDetectionVariant;
  if (enableHybridCycleDetection)
    hybridCycleDetectionVariant = std::true_type{};
  else
    hybridCycleDetectionVariant = std::false_type{};

  std::variant<std::true_type, std::false_type> lazyCycleDetectionVariant;
  if (enableLazyCycleDetection)
    lazyCycleDetectionVariant = std::true_type{};
  else
    lazyCycleDetectionVariant = std::false_type{};

  std::variant<std::true_type, std::false_type> differencePropagationVariant;
  if (enableDifferencePropagation)
    differencePropagationVariant = std::true_type{};
  else
    differencePropagationVariant = std::false_type{};

  std::variant<std::true_type, std::false_type> preferImplicitPropagationVariant;
  if (enablePreferImplicitPointees)
    preferImplicitPropagationVariant = std::true_type{};
  else
    preferImplicitPropagationVariant = std::false_type{};

  return std::visit(
      Dispatch,
      policyVariant,
      onlineCycleDetectionVariant,
      hybridCycleDetectionVariant,
      lazyCycleDetectionVariant,
      differencePropagationVariant,
      preferImplicitPropagationVariant);
}

const char *
PointerObjectConstraintSet::WorklistSolverPolicyToString(WorklistSolverPolicy policy)
{
  switch (policy)
  {
  case WorklistSolverPolicy::LeastRecentlyFired:
    return "LeastRecentlyFired";
  case WorklistSolverPolicy::TwoPhaseLeastRecentlyFired:
    return "TwoPhaseLeastRecentlyFired";
  case WorklistSolverPolicy::TopologicalSort:
    return "TopologicalSort";
  case WorklistSolverPolicy::FirstInFirstOut:
    return "FirstInFirstOut";
  case WorklistSolverPolicy::LastInFirstOut:
    return "LastInFirstOut";
  default:
    JLM_UNREACHABLE("Unknown WorklistSolverPolicy");
  }
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
  constraintClone->ConstraintSetFrozen_ = ConstraintSetFrozen_;
  return std::make_pair(std::move(setClone), std::move(constraintClone));
}

} // namespace jlm::llvm::aa

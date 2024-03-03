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

PointerObject::Index
PointerObjectSet::AddPointerObject(PointerObjectKind kind)
{
  JLM_ASSERT(PointerObjects_.size() < std::numeric_limits<PointerObject::Index>::max());
  PointerObjects_.emplace_back(kind);
  PointsToSets_.emplace_back(); // Add empty points-to-set
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
    count += pointerObject.GetKind() == kind;
  }
  return count;
}

PointerObject &
PointerObjectSet::GetPointerObject(PointerObject::Index index)
{
  JLM_ASSERT(index < NumPointerObjects());
  return PointerObjects_[index];
}

const PointerObject &
PointerObjectSet::GetPointerObject(PointerObject::Index index) const
{
  JLM_ASSERT(index < NumPointerObjects());
  return PointerObjects_[index];
}

PointerObject::Index
PointerObjectSet::CreateRegisterPointerObject(const rvsdg::output & rvsdgOutput)
{
  JLM_ASSERT(RegisterMap_.count(&rvsdgOutput) == 0);
  return RegisterMap_[&rvsdgOutput] = AddPointerObject(PointerObjectKind::Register);
}

PointerObject::Index
PointerObjectSet::GetRegisterPointerObject(const rvsdg::output & rvsdgOutput) const
{
  const auto it = RegisterMap_.find(&rvsdgOutput);
  JLM_ASSERT(it != RegisterMap_.end());
  return it->second;
}

std::optional<PointerObject::Index>
PointerObjectSet::TryGetRegisterPointerObject(const rvsdg::output & rvsdgOutput) const
{
  if (const auto it = RegisterMap_.find(&rvsdgOutput); it != RegisterMap_.end())
    return it->second;
  return std::nullopt;
}

void
PointerObjectSet::MapRegisterToExistingPointerObject(
    const rvsdg::output & rvsdgOutput,
    PointerObject::Index pointerObject)
{
  JLM_ASSERT(RegisterMap_.count(&rvsdgOutput) == 0);
  JLM_ASSERT(GetPointerObject(pointerObject).GetKind() == PointerObjectKind::Register);
  RegisterMap_[&rvsdgOutput] = pointerObject;
}

PointerObject::Index
PointerObjectSet::CreateDummyRegisterPointerObject()
{
  return AddPointerObject(PointerObjectKind::Register);
}

PointerObject::Index
PointerObjectSet::CreateAllocaMemoryObject(const rvsdg::node & allocaNode)
{
  JLM_ASSERT(AllocaMap_.count(&allocaNode) == 0);
  return AllocaMap_[&allocaNode] = AddPointerObject(PointerObjectKind::AllocaMemoryObject);
}

PointerObject::Index
PointerObjectSet::CreateMallocMemoryObject(const rvsdg::node & mallocNode)
{
  JLM_ASSERT(MallocMap_.count(&mallocNode) == 0);
  return MallocMap_[&mallocNode] = AddPointerObject(PointerObjectKind::MallocMemoryObject);
}

PointerObject::Index
PointerObjectSet::CreateGlobalMemoryObject(const delta::node & deltaNode)
{
  JLM_ASSERT(GlobalMap_.count(&deltaNode) == 0);
  return GlobalMap_[&deltaNode] = AddPointerObject(PointerObjectKind::GlobalMemoryObject);
}

PointerObject::Index
PointerObjectSet::CreateFunctionMemoryObject(const lambda::node & lambdaNode)
{
  JLM_ASSERT(!FunctionMap_.HasKey(&lambdaNode));
  const auto pointerObject = AddPointerObject(PointerObjectKind::FunctionMemoryObject);
  FunctionMap_.Insert(&lambdaNode, pointerObject);
  return pointerObject;
}

PointerObject::Index
PointerObjectSet::GetFunctionMemoryObject(const lambda::node & lambdaNode) const
{
  JLM_ASSERT(FunctionMap_.HasKey(&lambdaNode));
  return FunctionMap_.LookupKey(&lambdaNode);
}

const lambda::node &
PointerObjectSet::GetLambdaNodeFromFunctionMemoryObject(PointerObject::Index index) const
{
  JLM_ASSERT(FunctionMap_.HasValue(index));
  return *FunctionMap_.LookupValue(index);
}

PointerObject::Index
PointerObjectSet::CreateImportMemoryObject(const rvsdg::argument & importNode)
{
  JLM_ASSERT(ImportMap_.count(&importNode) == 0);
  return ImportMap_[&importNode] = AddPointerObject(PointerObjectKind::ImportMemoryObject);
}

const std::unordered_map<const rvsdg::output *, PointerObject::Index> &
PointerObjectSet::GetRegisterMap() const noexcept
{
  return RegisterMap_;
}

const std::unordered_map<const rvsdg::node *, PointerObject::Index> &
PointerObjectSet::GetAllocaMap() const noexcept
{
  return AllocaMap_;
}

const std::unordered_map<const rvsdg::node *, PointerObject::Index> &
PointerObjectSet::GetMallocMap() const noexcept
{
  return MallocMap_;
}

const std::unordered_map<const delta::node *, PointerObject::Index> &
PointerObjectSet::GetGlobalMap() const noexcept
{
  return GlobalMap_;
}

const util::BijectiveMap<const lambda::node *, PointerObject::Index> &
PointerObjectSet::GetFunctionMap() const noexcept
{
  return FunctionMap_;
}

const std::unordered_map<const rvsdg::argument *, PointerObject::Index> &
PointerObjectSet::GetImportMap() const noexcept
{
  return ImportMap_;
}

const util::HashSet<PointerObject::Index> &
PointerObjectSet::GetPointsToSet(PointerObject::Index idx) const
{
  JLM_ASSERT(idx < NumPointerObjects());
  return PointsToSets_[idx];
}

// Makes pointee a member of P(pointer)
bool
PointerObjectSet::AddToPointsToSet(PointerObject::Index pointer, PointerObject::Index pointee)
{
  JLM_ASSERT(pointer < NumPointerObjects());
  JLM_ASSERT(pointee < NumPointerObjects());
  // Assert the pointer object is a possible pointee
  JLM_ASSERT(GetPointerObject(pointee).CanBePointee());

  // If the pointer PointerObject can not point to anything, silently ignore
  if (!GetPointerObject(pointer).CanPoint())
    return false;

  return PointsToSets_[pointer].Insert(pointee);
}

// Makes P(superset) a superset of P(subset)
bool
PointerObjectSet::MakePointsToSetSuperset(
    PointerObject::Index superset,
    PointerObject::Index subset)
{
  JLM_ASSERT(superset < NumPointerObjects());
  JLM_ASSERT(subset < NumPointerObjects());

  // If the superset PointerObject can't point to anything, silently ignore
  if (!GetPointerObject(superset).CanPoint())
    return false;

  // If the superset PointerObject can't point to anything, silently ignore
  if (!GetPointerObject(superset).CanPoint())
    return false;

  auto & P_super = PointsToSets_[superset];
  const auto & P_sub = PointsToSets_[subset];

  bool modified = P_super.UnionWith(P_sub);

  // If the external node is in the subset, it must also be part of the superset
  if (GetPointerObject(subset).PointsToExternal())
    modified |= GetPointerObject(superset).MarkAsPointsToExternal();

  return modified;
}

// Markes all x in P(pointer) as escaped
bool
PointerObjectSet::MarkAllPointeesAsEscaped(PointerObject::Index pointer)
{
  bool modified = false;
  for (PointerObject::Index pointee : PointsToSets_[pointer].Items())
    modified |= GetPointerObject(pointee).MarkAsEscaped();

  return modified;
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
  for (PointerObject::Index x : set.GetPointsToSet(Pointer_).Items())
    modified |= set.MakePointsToSetSuperset(x, Value_);

  // If external in P(Pointer1_), P(external) should become a superset of P(Pointer2)
  // In practice, this means everything in P(Pointer2) escapes
  if (set.GetPointerObject(Pointer_).PointsToExternal())
    modified |= set.MarkAllPointeesAsEscaped(Value_);

  return modified;
}

// Make P(loaded) a superset of P(x) for all x in P(pointer)
bool
LoadConstraint::ApplyDirectly(PointerObjectSet & set)
{
  bool modified = false;
  for (PointerObject::Index x : set.GetPointsToSet(Pointer_).Items())
    modified |= set.MakePointsToSetSuperset(Value_, x);

  // P(pointer) "contains" external, then P(loaded) should also "contain" it
  if (set.GetPointerObject(Pointer_).PointsToExternal())
    modified |= set.GetPointerObject(Value_).MarkAsPointsToExternal();

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
    [[maybe_unused]] PointerObject::Index imported,
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
 *   void(PointerObject::Index superset, PointerObject::Index subset)
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
    PointerObject::Index lambda,
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

  const auto MakeSuperset = [&](PointerObject::Index superset, PointerObject::Index subset)
  {
    modified |= set.MakePointsToSetSuperset(superset, subset);
  };

  const auto MarkAsEscaped = [&](PointerObject::Index index)
  {
    modified |= set.GetPointerObject(index).MarkAsEscaped();
  };

  const auto MarkAsPointsToExternal = [&](PointerObject::Index index)
  {
    modified |= set.GetPointerObject(index).MarkAsPointsToExternal();
  };

  // For each possible function target, connect parameters and return values to the call node
  for (const auto target : set.GetPointsToSet(Pointer_).Items())
  {
    const auto kind = set.GetPointerObject(target).GetKind();
    if (kind == PointerObjectKind::ImportMemoryObject)
      HandleCallingImportedFunction(set, CallNode_, target, MarkAsEscaped, MarkAsPointsToExternal);
    else if (kind == PointerObjectKind::FunctionMemoryObject)
      HandleCallingLambdaFunction(set, CallNode_, target, MakeSuperset);
  }

  // If we might be calling an external function
  if (set.GetPointerObject(Pointer_).PointsToExternal())
    HandleCallingExternalFunction(set, CallNode_, MarkAsEscaped, MarkAsPointsToExternal);

  return modified;
}

bool
EscapeFlagConstraint::PropagateEscapedFlagsDirectly(PointerObjectSet & set)
{
  std::queue<PointerObject::Index> escapers;

  // First add all already escaped PointerObjects to the queue
  for (PointerObject::Index idx = 0; idx < set.NumPointerObjects(); idx++)
  {
    if (set.GetPointerObject(idx).HasEscaped())
      escapers.push(idx);
  }

  bool modified = false;

  // For all escapers, check if they point to any PointerObjects not marked as escaped
  while (!escapers.empty())
  {
    const PointerObject::Index escaper = escapers.front();
    escapers.pop();

    for (PointerObject::Index pointee : set.GetPointsToSet(escaper).Items())
    {
      if (set.GetPointerObject(pointee).MarkAsEscaped())
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
    PointerObject::Index lambda,
    MarkAsEscapedFunctor & markAsEscaped,
    MarkAsPointsToExternalFunctor & markAsPointsToExternal)
{
  JLM_ASSERT(set.GetPointerObject(lambda).GetKind() == PointerObjectKind::FunctionMemoryObject);
  JLM_ASSERT(set.GetPointerObject(lambda).HasEscaped());

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
    if (set.GetPointerObject(argumentPO.value()).PointsToExternal())
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
    if (set.GetPointerObject(resultPO.value()).HasEscaped())
      continue;

    // Mark the result register as escaped
    markAsEscaped(resultPO.value());
  }
}

bool
EscapedFunctionConstraint::PropagateEscapedFunctionsDirectly(PointerObjectSet & set)
{
  bool modified = false;

  const auto markAsEscaped = [&](PointerObject::Index index)
  {
    set.GetPointerObject(index).MarkAsEscaped();
    modified = true;
  };

  const auto markAsPointsToExternal = [&](PointerObject::Index index)
  {
    set.GetPointerObject(index).MarkAsPointsToExternal();
    modified = true;
  };

  for (const auto [lambda, lambdaPO] : set.GetFunctionMap())
  {
    if (set.GetPointerObject(lambdaPO).HasEscaped())
      HandleEscapedFunction(set, lambdaPO, markAsEscaped, markAsPointsToExternal);
  }

  return modified;
}

void
PointerObjectConstraintSet::AddPointerPointeeConstraint(
    PointerObject::Index pointer,
    PointerObject::Index pointee)
{
  // All set constraints are additive, so simple constraints like this can be directly applied.
  Set_.AddToPointsToSet(pointer, pointee);
}

void
PointerObjectConstraintSet::AddPointsToExternalConstraint(PointerObject::Index pointer)
{
  // Flags are never removed, so adding the flag now ensures it will be included.
  Set_.GetPointerObject(pointer).MarkAsPointsToExternal();
}

void
PointerObjectConstraintSet::AddRegisterContentEscapedConstraint(PointerObject::Index registerIndex)
{
  // Registers themselves can't escape in the classical sense, since they don't have an address.
  // (CanBePointee() is false)
  // When marked as Escaped, it instead means that the contents of the register has escaped.
  // This allows Escaped-flag propagation to mark any pointee the register might hold as escaped.
  auto & registerPointerObject = Set_.GetPointerObject(registerIndex);
  JLM_ASSERT(registerPointerObject.GetKind() == PointerObjectKind::Register);
  registerPointerObject.MarkAsEscaped();
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
  size_t numWorkItems = 0;

  // Create auxiliary superset graph.
  // If supersetEdges[x] contains y, (x -> y), that means P(y) supseteq P(x)
  std::vector<util::HashSet<PointerObject::Index>> supersetEdges(Set_.NumPointerObjects());

  // Create quick lookup tables for Load, Store and function call constraints.
  // Lookup is indexed by the constraint's pointer
  std::vector<std::vector<PointerObject::Index>> storeConstraints(Set_.NumPointerObjects());
  std::vector<std::vector<PointerObject::Index>> loadConstraints(Set_.NumPointerObjects());
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
  util::LrfWorklist<PointerObject::Index> worklist;
  for (PointerObject::Index i = 0; i < Set_.NumPointerObjects(); i++)
    worklist.PushWorkItem(i);

  // Helper function for adding superset edges, propagating everything currently in the subset.
  // The superset is added to the work list if its points-to set or flags are changed.
  const auto AddSupersetEdge = [&](PointerObject::Index superset, PointerObject::Index subset)
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
  const auto MarkAsEscaped = [&](PointerObject::Index index)
  {
    if (Set_.GetPointerObject(index).MarkAsEscaped())
      worklist.PushWorkItem(index);
  };

  // Helper function for flagging a pointer as pointing to external. Adds to the worklist if changed
  const auto MarkAsPointsToExternal = [&](PointerObject::Index index)
  {
    if (Set_.GetPointerObject(index).MarkAsPointsToExternal())
      worklist.PushWorkItem(index);
  };

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
      if (Set_.GetPointerObject(n).PointsToExternal())
        MarkAsEscaped(value);
    }

    // Loads on the form value = *n.
    for (const auto value : loadConstraints[n])
    {
      // This loop ensures P(value) supseteq *P(n)
      for (const auto pointee : Set_.GetPointsToSet(n).Items())
        AddSupersetEdge(value, pointee);

      // If P(n) contains "external", the loaded value may also point to external
      if (Set_.GetPointerObject(n).PointsToExternal())
        MarkAsPointsToExternal(value);
    }

    // Function calls on the form (*n)()
    for (const auto callNode : callConstraints[n])
    {
      // Connect the inputs and outputs of the callNode to every possible function pointee
      for (const auto pointee : Set_.GetPointsToSet(n).Items())
      {
        const auto kind = Set_.GetPointerObject(pointee).GetKind();
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
      if (Set_.GetPointerObject(n).PointsToExternal())
        HandleCallingExternalFunction(Set_, *callNode, MarkAsEscaped, MarkAsPointsToExternal);
    }

    // Propagate P(n) along all edges n -> superset
    for (const auto superset : supersetEdges[n].Items())
    {
      if (Set_.MakePointsToSetSuperset(superset, n))
        worklist.PushWorkItem(superset);
    }

    // If escaped, propagate escaped flag to all pointees
    if (Set_.GetPointerObject(n).HasEscaped())
    {
      for (const auto pointee : Set_.GetPointsToSet(n).Items())
        MarkAsEscaped(pointee);

      // Escaped functions also need to flag arguments and results in the function body
      if (Set_.GetPointerObject(n).GetKind() == PointerObjectKind::FunctionMemoryObject)
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

} // namespace jlm::llvm::aa

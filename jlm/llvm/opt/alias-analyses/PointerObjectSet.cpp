/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>

#include <jlm/llvm/ir/operators/call.hpp>

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

// P(superset) is a superset of P(subset)
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

// For escaped functions, the result must be marked as escaped,
// and all arguments of pointer type as pointing to external.
bool
HandleEscapingFunctionConstraint::ApplyDirectly(PointerObjectSet & set)
{
  if (EscapeHandled_ || !set.GetPointerObject(Lambda_).HasEscaped())
  {
    return false;
  }
  EscapeHandled_ = true;

  // We now go though the lambda's inner region and apply the necessary flags
  auto & lambdaNode = set.GetLambdaNodeFromFunctionMemoryObject(Lambda_);

  // All the function's arguments need to be flagged as PointsToExternal
  for (auto & argument : lambdaNode.fctarguments())
  {
    if (!is<PointerType>(argument.type()))
      continue;

    const auto argumentPO = set.GetRegisterPointerObject(argument);
    set.GetPointerObject(argumentPO).MarkAsPointsToExternal();
  }

  // All results of pointer type need to be flagged as HasEscaped
  for (auto & result : lambdaNode.fctresults())
  {
    if (!is<PointerType>(result.type()))
      continue;

    // Mark the register as escaped, which will propagate the escaped flag to all pointees
    const auto resultPO = set.GetRegisterPointerObject(*result.origin());
    set.GetPointerObject(resultPO).MarkAsEscaped();
  }

  return true;
}

// When a function call's callee can be anything visible outside of the module
bool
FunctionCallConstraint::HandleCallingExternalFunction(PointerObjectSet & set)
{
  bool modified = false;

  // Mark all the call's inputs as escaped, and all the outputs as pointing to external
  for (size_t n = 0; n < CallNode_.NumArguments(); n++)
  {
    const auto & inputRegister = *CallNode_.Argument(n)->origin();
    if (!is<PointerType>(inputRegister.type()))
      continue;

    const auto inputRegisterPO = set.GetRegisterPointerObject(inputRegister);
    modified |= set.GetPointerObject(inputRegisterPO).MarkAsEscaped();
  }

  for (size_t n = 0; n < CallNode_.NumResults(); n++)
  {
    const auto & outputRegister = *CallNode_.Result(n);
    if (!is<PointerType>(outputRegister.type()))
      continue;

    const auto outputRegisterPO = set.GetRegisterPointerObject(outputRegister);
    modified |= set.GetPointerObject(outputRegisterPO).MarkAsPointsToExternal();
  }

  return modified;
}

// When a function call's callee might be a given function imported from outside the module
bool
FunctionCallConstraint::HandleCallingImportedFunction(
    PointerObjectSet & set,
    PointerObject::Index imported)
{
  // FIXME: Add special handling of common library functions
  // Otherwise we don't know anything about the function
  return HandleCallingExternalFunction(set);
}

// When a function call's callee might be a lambda node in our module
bool
FunctionCallConstraint::HandleCallingLambdaFunction(
    PointerObjectSet & set,
    PointerObject::Index lambda)
{
  bool modified = false;

  auto & lambdaNode = set.GetLambdaNodeFromFunctionMemoryObject(lambda);

  // If the number of parameters or number of results doesn't line up,
  // assume this is not the function we are calling.
  // Note that the number of arguments and results include 3 state edges: memory, loop and IO.
  // Varargs are properly handled, since they get merged by a valist_op node before the CallNode.
  if (lambdaNode.nfctarguments() != CallNode_.NumArguments()
      || lambdaNode.nfctresults() != CallNode_.NumResults())
    return false;

  // Pass all call node inputs to the function's subregion
  for (size_t n = 0; n < CallNode_.NumArguments(); n++)
  {
    const auto & inputRegister = *CallNode_.Argument(n)->origin();
    const auto & argumentRegister = *lambdaNode.fctargument(n);
    if (!is<PointerType>(inputRegister.type()) || !is<PointerType>(argumentRegister.type()))
      continue;

    const auto inputRegisterPO = set.GetRegisterPointerObject(inputRegister);
    const auto argumentRegisterPO = set.GetRegisterPointerObject(argumentRegister);
    modified |= set.MakePointsToSetSuperset(argumentRegisterPO, inputRegisterPO);
  }

  // Pass the function's subregion results to the output of the call node
  for (size_t n = 0; n < CallNode_.NumResults(); n++)
  {
    const auto & outputRegister = *CallNode_.Result(n);
    const auto & resultRegister = *lambdaNode.fctresult(n)->origin();
    if (!is<PointerType>(outputRegister.type()) || !is<PointerType>(resultRegister.type()))
      continue;

    const auto outputRegisterPO = set.GetRegisterPointerObject(outputRegister);
    const auto resultRegisterPO = set.GetRegisterPointerObject(resultRegister);
    modified |= set.MakePointsToSetSuperset(outputRegisterPO, resultRegisterPO);
  }

  return modified;
};

// Connects function calls to every possible target function
bool
FunctionCallConstraint::ApplyDirectly(PointerObjectSet & set)
{
  bool modified = false;

  // For each possible function target, connect parameters and return values to the call node
  for (const auto target : set.GetPointsToSet(CallTarget_).Items())
  {
    const auto kind = set.GetPointerObject(target).GetKind();
    if (kind == PointerObjectKind::ImportMemoryObject)
      modified |= HandleCallingImportedFunction(set, target);
    else if (kind == PointerObjectKind::FunctionMemoryObject)
      modified |= HandleCallingLambdaFunction(set, target);
  }

  // If we might be calling an external function
  if (set.GetPointerObject(CallTarget_).PointsToExternal())
  {
    modified |= HandleCallingExternalFunction(set);
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

    modified |= PropagateEscapedFlag();
  }

  return numIterations;
}

bool
PointerObjectConstraintSet::PropagateEscapedFlag()
{
  std::queue<PointerObject::Index> escapers;

  // First add all already escaped PointerObjects to the queue
  for (PointerObject::Index idx = 0; idx < Set_.NumPointerObjects(); idx++)
  {
    if (Set_.GetPointerObject(idx).HasEscaped())
      escapers.push(idx);
  }

  bool modified = false;

  // For all escapers, check if they point to any PointerObjects not marked as escaped
  while (!escapers.empty())
  {
    const PointerObject::Index escaper = escapers.front();
    escapers.pop();

    for (PointerObject::Index pointee : Set_.GetPointsToSet(escaper).Items())
    {
      if (Set_.GetPointerObject(pointee).MarkAsEscaped())
      {
        // Add the newly marked PointerObject to the queue, in case the flag can be propagated
        escapers.push(pointee);
        modified = true;
      }
    }
  }

  return modified;
}

} // namespace jlm::llvm::aa

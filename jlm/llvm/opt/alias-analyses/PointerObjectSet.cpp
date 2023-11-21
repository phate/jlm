/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>

#include <queue>

namespace jlm::llvm::aa
{

PointerObject::Index
PointerObjectSet::AddPointerObject(PointerObjectKind kind)
{
  PointerObjects_.emplace_back(kind);
  PointsToSets_.emplace_back(); // Add empty points-to-set
  return PointerObjects_.size() - 1;
}

PointerObject::Index
PointerObjectSet::CreateRegisterPointerObject(const rvsdg::output & rvsdgOutput)
{
  JLM_ASSERT(RegisterMap_.count(&rvsdgOutput) == 0);
  return RegisterMap_[&rvsdgOutput] = AddPointerObject(PointerObjectKind::Register);
}

void
PointerObjectSet::MapRegisterToExistingPointerObject(
    const rvsdg::output & rvsdgOutput,
    PointerObject::Index pointerObject)
{
  JLM_ASSERT(RegisterMap_.count(&rvsdgOutput) == 0);
  JLM_ASSERT(pointerObject < NumPointerObjects());
  JLM_ASSERT(GetPointerObject(pointerObject).GetKind() == PointerObjectKind::Register);
  RegisterMap_[&rvsdgOutput] = pointerObject;
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
  JLM_ASSERT(FunctionMap_.count(&lambdaNode) == 0);
  return FunctionMap_[&lambdaNode] = AddPointerObject(PointerObjectKind::FunctionMemoryObject);
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

const std::unordered_map<const lambda::node *, PointerObject::Index> &
PointerObjectSet::GetFunctionMap() const noexcept
{
  return FunctionMap_;
}

const std::unordered_map<const rvsdg::argument *, PointerObject::Index> &
PointerObjectSet::GetImportMap() const noexcept
{
  return ImportMap_;
}

PointerObject::Index
PointerObjectSet::NumPointerObjects() const noexcept
{
  return PointerObjects_.size();
}

PointerObject &
PointerObjectSet::GetPointerObject(PointerObject::Index index)
{
  JLM_ASSERT(index <= NumPointerObjects());
  return PointerObjects_[index];
}

const PointerObject &
PointerObjectSet::GetPointerObject(PointerObject::Index index) const
{
  JLM_ASSERT(index <= NumPointerObjects());
  return PointerObjects_[index];
}

const util::HashSet<PointerObject::Index> &
PointerObjectSet::GetPointsToSet(PointerObject::Index idx) const
{
  JLM_ASSERT(idx <= NumPointerObjects());
  return PointsToSets_[idx];
}

// Makes pointee a member of P(pointer)
bool
PointerObjectSet::AddToPointsToSet(PointerObject::Index pointer, PointerObject::Index pointee)
{
  JLM_ASSERT(pointer < NumPointerObjects());
  JLM_ASSERT(pointee < NumPointerObjects());
  // Registers can not be pointed to
  JLM_ASSERT(GetPointerObject(pointee).GetKind() != PointerObjectKind::Register);

  return PointsToSets_[pointer].Insert(pointee);
}

// Makes P(superset) a superset of P(subset)
bool
PointerObjectSet::MakePointsToSetSuperset(
    PointerObject::Index superset,
    PointerObject::Index subset)
{
  JLM_ASSERT(superset <= NumPointerObjects());
  JLM_ASSERT(subset <= NumPointerObjects());

  auto & P_super = PointsToSets_[superset];
  auto & P_sub = PointsToSets_[subset];

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
SupersetConstraint::Apply(PointerObjectSet & set)
{
  return set.MakePointsToSetSuperset(Superset_, Subset_);
}

// for all x in P(pointer1), make P(x) a superset of P(pointer2)
bool
AllPointeesPointToSupersetConstraint::Apply(PointerObjectSet & set)
{
  bool modified = false;
  for (PointerObject::Index x : set.GetPointsToSet(Pointer1_).Items())
    modified |= set.MakePointsToSetSuperset(x, Pointer2_);

  // If external in P(Pointer1_), P(external) should become a superset of P(Pointer2)
  // In practice, this means everything in P(Pointer2) escapes
  if (set.GetPointerObject(Pointer1_).PointsToExternal())
    modified |= set.MarkAllPointeesAsEscaped(Pointer2_);

  return modified;
}

// Make P(loaded) a superset of P(x) for all x in P(pointer)
bool
SupersetOfAllPointeesConstraint::Apply(PointerObjectSet & set)
{
  bool modified = false;
  for (PointerObject::Index x : set.GetPointsToSet(Pointer_).Items())
    modified |= set.MakePointsToSetSuperset(Loaded_, x);

  // Handling pointing to external is done by MakePointsToSetSuperset,
  // Propagating escaped status is handled by different constraints

  return modified;
}

void
PointerObjectConstraintSet::AddPointerPointeeConstraint(
    PointerObject::Index pointer,
    PointerObject::Index pointee)
{
  // All set constraints are additive, so simple constraints like this can be directly applied and
  // forgotten.
  Set_.AddToPointsToSet(pointer, pointee);
}

void
PointerObjectConstraintSet::AddPointsToExternalConstraint(PointerObject::Index pointer)
{
  // Flags are never removed, so adding the flag now ensures it will be included in the final
  // solution
  Set_.GetPointerObject(pointer).MarkAsPointsToExternal();
}

void
PointerObjectConstraintSet::AddRegisterContentEscapedConstraint(PointerObject::Index registerIndex)
{
  // Registers themselves can't really escape, since they don't have an address
  // We can however mark it as escaped, and let escape flag propagation ensure everything it ever
  // points to is marked.
  auto & registerPointerObject = Set_.GetPointerObject(registerIndex);
  JLM_ASSERT(registerPointerObject.GetKind() == PointerObjectKind::Register);
  registerPointerObject.MarkAsEscaped();
}

void
PointerObjectConstraintSet::AddConstraint(ConstraintVariant c)
{
  Constraints_.push_back(c);
}

void
PointerObjectConstraintSet::Solve()
{
  // Keep applying constraints until no sets are modified
  bool modified = true;

  while (modified)
  {
    modified = false;

    for (auto & constraint : Constraints_)
    {
      std::visit(
          [&](auto & constraint)
          {
            modified |= constraint.Apply(Set_);
          },
          constraint);
    }

    modified |= PropagateEscapedFlag();
  }
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


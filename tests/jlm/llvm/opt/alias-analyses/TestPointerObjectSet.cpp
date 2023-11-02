/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>

#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>

#include <cassert>

static void
TestFlagFunctions()
{
  using namespace jlm::llvm::aa;

  PointerObject object(PointerObjectKind::Register);
  assert(object.GetKind() == PointerObjectKind::Register);

  // Important ordering. Escaped implies PointsToExternal
  assert(!object.PointsToExternal());
  assert(object.MarkAsPointsToExternal());
  assert(object.PointsToExternal());
  assert(!object.MarkAsPointsToExternal());
  assert(object.PointsToExternal());

  assert(!object.HasEscaped());
  assert(object.MarkAsEscaped());
  assert(object.HasEscaped());
  assert(!object.MarkAsEscaped());
  assert(object.HasEscaped());

  // Test that Escaped implies PointsToExternal
  object = PointerObject(PointerObjectKind::AllocaMemoryObject);
  assert(object.GetKind() == PointerObjectKind::AllocaMemoryObject);
  assert(!object.PointsToExternal());
  assert(object.MarkAsEscaped());
  assert(object.PointsToExternal());
  assert(!object.MarkAsPointsToExternal());
}

static void
TestAddToPointsToSet()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(1);

  PointerObjectSet set;
  auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));

  assert(set.GetPointsToSet(reg0).size() == 0);

  assert(set.AddToPointsToSet(reg0, alloca0));
  assert(set.GetPointsToSet(reg0).size() == 1);
  assert(set.GetPointsToSet(reg0).count(alloca0));

  // Trying to add it again returns false
  assert(!set.AddToPointsToSet(reg0, alloca0));
}

static void
TestMakePointsToSetSuperset()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(3);

  PointerObjectSet set;
  auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1));
  auto reg1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));
  auto alloca2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2));

  set.AddToPointsToSet(reg0, alloca0);
  set.AddToPointsToSet(reg1, alloca1);

  assert(set.GetPointsToSet(reg0).size() == 1);
  assert(set.GetPointsToSet(reg0).count(alloca0));

  assert(set.MakePointsToSetSuperset(reg0, reg1));
  assert(set.GetPointsToSet(reg1).size() == 1);
  assert(set.GetPointsToSet(reg0).size() == 2);
  assert(set.GetPointsToSet(reg0).count(alloca1));

  // Calling it again without changing reg1 makes no difference, returns false
  assert(!set.MakePointsToSetSuperset(reg0, reg1));

  // Add an additional member to P(reg1)
  set.AddToPointsToSet(reg1, alloca2);
  assert(set.MakePointsToSetSuperset(reg0, reg1));
  assert(set.GetPointsToSet(reg0).count(alloca2));
}

static void
TestMarkAllPointeesAsEscaped()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(2);

  PointerObjectSet set;
  auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1));

  set.AddToPointsToSet(reg0, alloca0);
  set.AddToPointsToSet(reg0, alloca1);
  assert(set.MarkAllPointeesAsEscaped(reg0));

  assert(set.GetPointerObject(alloca0).HasEscaped());
  assert(!set.GetPointerObject(alloca1).HasEscaped());
}

static void
TestSupersetConstraint()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(3);

  PointerObjectSet set;
  auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1));
  auto reg1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));
  auto alloca2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2));
  auto reg2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2));

  set.AddToPointsToSet(reg0, alloca0);
  set.AddToPointsToSet(reg1, alloca1);
  set.AddToPointsToSet(reg2, alloca2);

  SupersetConstraint c1(alloca0, reg1);
  while(c1.Apply(set));
  assert(set.GetPointsToSet(alloca0).count(alloca1));

  SupersetConstraint c2(reg1, reg2);
  while(c2.Apply(set));
  assert(set.GetPointsToSet(reg1).count(alloca2));

  // Apply c1 again
  while(c1.Apply(set));
  assert(set.GetPointsToSet(alloca0).count(alloca2));
}

static int
TestPointerObjectSet()
{
  TestFlagFunctions();
  TestAddToPointsToSet();
  TestMakePointsToSetSuperset();
  TestMarkAllPointeesAsEscaped();
  TestSupersetConstraint();
  return 0;
}

JLM_UNIT_TEST_REGISTER(
"jlm/llvm/opt/alias-analyses/TestPointerObjectSet",
TestPointerObjectSet)

/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>

#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>

#include <cassert>

// Test the flag functions on the PointerObject class
static void
TestFlagFunctions()
{
  using namespace jlm::llvm::aa;

  PointerObject object(PointerObjectKind::Register);
  assert(object.GetKind() == PointerObjectKind::Register);

  // Important ordering. Test PointsToExternal first, since it is implied by Escaped.
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

// Test creating pointer objects for each type of memory node
static void
TestCreatePointerObjects()
{
  using namespace jlm::llvm::aa;

  jlm::tests::AllMemoryNodesTest rvsdg;
  rvsdg.EnsureInitialized();

  PointerObjectSet set;
  auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode());
  auto malloc0 = set.CreateMallocMemoryObject(rvsdg.GetMallocNode());
  auto delta0 = set.CreateGlobalMemoryObject(rvsdg.GetDeltaNode());
  auto lambda0 = set.CreateFunctionMemoryObject(rvsdg.GetLambdaNode());
  auto import0 = set.CreateImportMemoryObject(rvsdg.GetImportOutput());

  assert(set.GetPointerObject(alloca0).GetKind() == PointerObjectKind::AllocaMemoryObject);
  assert(set.GetPointerObject(malloc0).GetKind() == PointerObjectKind::MallocMemoryObject);
  assert(set.GetPointerObject(delta0).GetKind() == PointerObjectKind::GlobalMemoryObject);
  assert(set.GetPointerObject(lambda0).GetKind() == PointerObjectKind::FunctionMemoryObject);
  assert(set.GetPointerObject(import0).GetKind() == PointerObjectKind::ImportMemoryObject);

  // Imported objects should have been marked as escaped
  assert(set.GetPointerObject(import0).HasEscaped());
}

// Test the PointerObjectSet method for adding pointer objects to another pointer object's points-to-set
static void
TestAddToPointsToSet()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(1);
  rvsdg.EnsureInitialized();

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

// Test the PointerObjectSet method for making one points-to-set a superset of another
static void
TestMakePointsToSetSuperset()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(3);
  rvsdg.EnsureInitialized();

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

// Test the PointerObjectSet method for marking all pointees of the given pointer as escaped
static void
TestMarkAllPointeesAsEscaped()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(3);
  rvsdg.EnsureInitialized();

  PointerObjectSet set;
  auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1));
  auto alloca2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2));

  set.AddToPointsToSet(reg0, alloca0);
  set.AddToPointsToSet(reg0, alloca1);
  assert(set.MarkAllPointeesAsEscaped(reg0));

  assert(set.GetPointerObject(alloca0).HasEscaped());
  assert(set.GetPointerObject(alloca1).HasEscaped());
  assert(!set.GetPointerObject(reg0).HasEscaped());
  assert(!set.GetPointerObject(alloca2).HasEscaped());
}

// Test the SupersetConstraint's Apply function
static void
TestSupersetConstraint()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(3);
  rvsdg.EnsureInitialized();

  PointerObjectSet set;
  auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1));
  auto reg1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));
  auto alloca2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2));
  auto reg2 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(2));

  set.AddToPointsToSet(reg0, alloca0);
  set.AddToPointsToSet(reg1, alloca1);
  set.AddToPointsToSet(reg2, alloca2);

  // Make alloca0 point to everything reg1 points to
  SupersetConstraint c1(alloca0, reg1);
  assert(c1.Apply(set));
  while(c1.Apply(set));
  // For now this only makes alloca0 point to alloca1
  assert(set.GetPointsToSet(alloca0).size() == 1);
  assert(set.GetPointsToSet(alloca0).count(alloca1) == 1);

  // Make reg1 point to everything reg2 points to
  SupersetConstraint c2(reg1, reg2);
  assert(c2.Apply(set));
  while(c2.Apply(set));
  // This makes alloca2 a member of P(reg1)
  assert(set.GetPointsToSet(reg1).count(alloca2));

  // Apply c1 again
  assert(c1.Apply(set));
  while(c1.Apply(set));
  // Now alloca0 should also point to alloca2
  assert(set.GetPointsToSet(alloca0).size() == 2);
  assert(set.GetPointsToSet(alloca0).count(alloca2));

  // Make reg2 point to external, and propagate through constraints
  set.GetPointerObject(reg2).MarkAsPointsToExternal();
  assert(c2.Apply(set));
  while(c2.Apply(set));
  assert(c1.Apply(set));
  while(c1.Apply(set));
  // Now both reg1 and alloca0 may point to external
  assert(set.GetPointerObject(reg1).PointsToExternal());
  assert(set.GetPointerObject(alloca0).PointsToExternal());
}

// Test the AllPointeesPointToSupersetConstraint's Apply function
static void
TestAllPointeesPointToSupersetConstraint()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(3);
  rvsdg.EnsureInitialized();

  PointerObjectSet set;
  auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1));
  auto reg1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));
  auto alloca2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2));
  auto reg2 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(2));

  set.AddToPointsToSet(reg0, alloca0);
  set.AddToPointsToSet(reg1, alloca1);
  set.AddToPointsToSet(reg2, alloca2);

  // Add *alloca0 = reg2, which will do nothing, since alloca0 can't be pointing to anything yet
  AllPointeesPointToSupersetConstraint c1(alloca0, reg2);
  bool modified1 = c1.Apply(set);
  assert(!modified1);

  // This means *reg0 = reg1, and as we know, reg0 points to alloca0
  // This should make alloca0 point to anything reg1 points to, aka alloca1
  AllPointeesPointToSupersetConstraint c2(reg0, reg1);
  assert(c2.Apply(set));
  while(c2.Apply(set));
  assert(set.GetPointsToSet(alloca0).size() == 1);
  assert(set.GetPointsToSet(alloca0).count(alloca1) == 1);

  // Do c1 again, now that alloca0 points to alloca1
  assert(c1.Apply(set));
  while(c1.Apply(set));
  assert(set.GetPointsToSet(alloca1).size() == 1);
  assert(set.GetPointsToSet(alloca1).count(alloca2) == 1);
}

// Test the SupersetOfAllPointeesConstraint's Apply function
static void
TestSupersetOfAllPointeesConstraint()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(3);
  rvsdg.EnsureInitialized();

  PointerObjectSet set;
  auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1));
  auto reg1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));
  auto alloca2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2));
  auto reg2 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(2));

  set.AddToPointsToSet(reg0, alloca0);
  set.AddToPointsToSet(reg1, alloca1);
  set.AddToPointsToSet(reg2, alloca2);

  // Makes reg1 = *reg0, where reg0 currently just points to alloca0
  // Since alloca0 currently has no pointees, this does nothing
  SupersetOfAllPointeesConstraint c1(reg1, reg0);
  bool modified1 = c1.Apply(set);
  assert(!modified1);

  // Make alloca0 point to something: alloca2
  set.AddToPointsToSet(alloca0, alloca2);
  // The constraint now makes a difference
  assert(c1.Apply(set));
  while(c1.Apply(set));
  assert(set.GetPointsToSet(reg1).size() == 2);
  assert(set.GetPointsToSet(reg1).count(alloca2) == 1);
}

static int
TestPointerObjectSet()
{
  TestFlagFunctions();
  TestCreatePointerObjects();
  TestAddToPointsToSet();
  TestMakePointsToSetSuperset();
  TestMarkAllPointeesAsEscaped();
  TestSupersetConstraint();
  TestAllPointeesPointToSupersetConstraint();
  TestSupersetOfAllPointeesConstraint();
  return 0;
}

JLM_UNIT_TEST_REGISTER(
"jlm/llvm/opt/alias-analyses/TestPointerObjectSet",
TestPointerObjectSet)

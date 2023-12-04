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

  // Test that Escaped implies PointsToExternal, for memory objects
  object = PointerObject(PointerObjectKind::AllocaMemoryObject);
  assert(object.GetKind() == PointerObjectKind::AllocaMemoryObject);
  assert(!object.PointsToExternal());
  assert(object.MarkAsEscaped());
  assert(object.PointsToExternal());
  assert(!object.MarkAsPointsToExternal());

  // Test that Escaped does not imply PointsToExternal, for registers (CanBePointee() == false)
  object = PointerObject(PointerObjectKind::Register);
  object.MarkAsEscaped();
  assert(!object.PointsToExternal());

  // Test that Functions, who have CanPoint() == false, can not be made to PointToExternal
  object = PointerObject(PointerObjectKind::FunctionMemoryObject);
  assert(!object.MarkAsPointsToExternal());
  object.MarkAsEscaped();
  assert(!object.PointsToExternal());
}

// Test creating pointer objects for each type of memory node
static void
TestCreatePointerObjects()
{
  using namespace jlm::llvm::aa;

  jlm::tests::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  PointerObjectSet set;

  // Register PointerObjects have some extra ways of being created: Dummy and mapping
  auto register0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput());
  set.MapRegisterToExistingPointerObject(rvsdg.GetDeltaOutput(), register0);
  auto dummy0 = set.CreateDummyRegisterPointerObject();

  // For PointerObjects representing MemoryObjects, there is only one Create function
  auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode());
  auto malloc0 = set.CreateMallocMemoryObject(rvsdg.GetMallocNode());
  auto delta0 = set.CreateGlobalMemoryObject(rvsdg.GetDeltaNode());
  auto lambda0 = set.CreateFunctionMemoryObject(rvsdg.GetLambdaNode());
  auto import0 = set.CreateImportMemoryObject(rvsdg.GetImportOutput());

  assert(set.GetPointerObject(register0).GetKind() == PointerObjectKind::Register);
  assert(set.GetPointerObject(dummy0).GetKind() == PointerObjectKind::Register);
  assert(set.GetPointerObject(alloca0).GetKind() == PointerObjectKind::AllocaMemoryObject);
  assert(set.GetPointerObject(malloc0).GetKind() == PointerObjectKind::MallocMemoryObject);
  assert(set.GetPointerObject(delta0).GetKind() == PointerObjectKind::GlobalMemoryObject);
  assert(set.GetPointerObject(lambda0).GetKind() == PointerObjectKind::FunctionMemoryObject);
  assert(set.GetPointerObject(import0).GetKind() == PointerObjectKind::ImportMemoryObject);

  // Registers have helper function for looking up existing PointerObjects
  assert(set.GetRegisterPointerObject(rvsdg.GetAllocaOutput()) == register0);
  assert(set.GetRegisterPointerObject(rvsdg.GetDeltaOutput()) == register0);

  // Funtions have the same, but also in the other direction
  assert(set.GetFunctionMemoryObject(rvsdg.GetLambdaNode()) == lambda0);
  assert(&set.GetLambdaNodeFromFunctionMemoryObject(lambda0) == &rvsdg.GetLambdaNode());

  // The maps can also be accessed directly
  assert(set.GetRegisterMap().at(&rvsdg.GetAllocaOutput()) == register0);
  assert(set.GetRegisterMap().at(&rvsdg.GetDeltaOutput()) == register0);
  assert(set.GetAllocaMap().at(&rvsdg.GetAllocaNode()) == alloca0);
  assert(set.GetMallocMap().at(&rvsdg.GetMallocNode()) == malloc0);
  assert(set.GetGlobalMap().at(&rvsdg.GetDeltaNode()) == delta0);
  assert(set.GetFunctionMap().at(&rvsdg.GetLambdaNode()) == lambda0);
  assert(set.GetImportMap().at(&rvsdg.GetImportOutput()) == import0);

  // Imported objects should have been marked as escaped
  assert(set.GetPointerObject(import0).HasEscaped());

  // Registers should be marked as !CanBePointee()
  assert(!set.GetPointerObject(register0).CanBePointee());

  // Functions should be marked as !CanPoint()
  assert(!set.GetPointerObject(lambda0).CanPoint());
}

// Test the PointerObjectSet method for adding pointer objects to another pointer object's
// points-to-set
static void
TestAddToPointsToSet()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(1);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  const auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));

  assert(set.GetPointsToSet(reg0).Size() == 0);

  assert(set.AddToPointsToSet(reg0, alloca0));
  assert(set.GetPointsToSet(reg0).Size() == 1);
  assert(set.GetPointsToSet(reg0).Contains(alloca0));

  // Trying to add it again returns false
  assert(!set.AddToPointsToSet(reg0, alloca0));

  // Trying to make a function (CanPoint() == false) point to something is a no-op
  const auto function0 = set.CreateFunctionMemoryObject(rvsdg.GetFunction());
  assert(!set.AddToPointsToSet(function0, alloca0));
  assert(set.GetPointsToSet(function0).Size() == 0);
}

// Test the PointerObjectSet method for making one points-to-set a superset of another
static void
TestMakePointsToSetSuperset()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(3);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  const auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  const auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1));
  const auto reg1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));
  const auto alloca2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2));

  set.AddToPointsToSet(reg0, alloca0);
  set.AddToPointsToSet(reg1, alloca1);

  assert(set.GetPointsToSet(reg0).Size() == 1);
  assert(set.GetPointsToSet(reg0).Contains(alloca0));

  assert(set.MakePointsToSetSuperset(reg0, reg1));
  assert(set.GetPointsToSet(reg1).Size() == 1);
  assert(set.GetPointsToSet(reg0).Size() == 2);
  assert(set.GetPointsToSet(reg0).Contains(alloca1));

  // Calling it again without changing reg1 makes no difference, returns false
  assert(!set.MakePointsToSetSuperset(reg0, reg1));

  // Add an additional member to P(reg1)
  set.AddToPointsToSet(reg1, alloca2);
  assert(set.MakePointsToSetSuperset(reg0, reg1));
  assert(set.GetPointsToSet(reg0).Contains(alloca2));

  // Trying to make a function's points-to-set a superset is a no-op
  // Since functions have CanPoint() == false.
  const auto function0 = set.CreateFunctionMemoryObject(rvsdg.GetFunction());
  assert(!set.MakePointsToSetSuperset(function0, reg0));
  assert(set.GetPointsToSet(function0).Size() == 0);
}

// Test the PointerObjectSet method for marking all pointees of the given pointer as escaped
static void
TestMarkAllPointeesAsEscaped()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(3);
  rvsdg.InitializeTest();

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
  rvsdg.InitializeTest();

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
  while (c1.Apply(set))
    ;
  // For now this only makes alloca0 point to alloca1
  assert(set.GetPointsToSet(alloca0).Size() == 1);
  assert(set.GetPointsToSet(alloca0).Contains(alloca1));

  // Make reg1 point to everything reg2 points to
  SupersetConstraint c2(reg1, reg2);
  assert(c2.Apply(set));
  while (c2.Apply(set))
    ;
  // This makes alloca2 a member of P(reg1)
  assert(set.GetPointsToSet(reg1).Contains(alloca2));

  // Apply c1 again
  assert(c1.Apply(set));
  while (c1.Apply(set))
    ;
  // Now alloca0 should also point to alloca2
  assert(set.GetPointsToSet(alloca0).Size() == 2);
  assert(set.GetPointsToSet(alloca0).Contains(alloca2));

  // Make reg2 point to external, and propagate through constraints
  set.GetPointerObject(reg2).MarkAsPointsToExternal();
  assert(c2.Apply(set));
  while (c2.Apply(set))
    ;
  assert(c1.Apply(set));
  while (c1.Apply(set))
    ;
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
  rvsdg.InitializeTest();

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
  assert(!c1.Apply(set));

  // This means *reg0 = reg1, and as we know, reg0 points to alloca0
  // This should make alloca0 point to anything reg1 points to, aka alloca1
  AllPointeesPointToSupersetConstraint c2(reg0, reg1);
  assert(c2.Apply(set));
  while (c2.Apply(set))
    ;
  assert(set.GetPointsToSet(alloca0).Size() == 1);
  assert(set.GetPointsToSet(alloca0).Contains(alloca1));

  // Do c1 again, now that alloca0 points to alloca1
  assert(c1.Apply(set));
  while (c1.Apply(set))
    ;
  assert(set.GetPointsToSet(alloca1).Size() == 1);
  assert(set.GetPointsToSet(alloca1).Contains(alloca2));
}

// Test the SupersetOfAllPointeesConstraint's Apply function
static void
TestSupersetOfAllPointeesConstraint()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(3);
  rvsdg.InitializeTest();

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
  assert(!c1.Apply(set));

  // Make alloca0 point to something: alloca2
  set.AddToPointsToSet(alloca0, alloca2);
  // The constraint now makes a difference
  assert(c1.Apply(set));
  while (c1.Apply(set))
    ;
  assert(set.GetPointsToSet(reg1).Size() == 2);
  assert(set.GetPointsToSet(reg1).Contains(alloca2));
}

static void
TestHandleEscapingFunctionConstraint()
{
  using namespace jlm::llvm::aa;

  jlm::tests::EscapingLocalFunctionTest rvsdg;
  rvsdg.InitializeTest();

  const auto & localFunction = rvsdg.GetLocalFunction();
  const auto & localFunctionRegister = rvsdg.GetLocalFunctionRegister();
  const auto & exportedFunction = rvsdg.GetExportedFunction();
  const auto & exportedFunctionReturn = *exportedFunction.fctresult(0)->origin();

  PointerObjectSet set;
  const auto localFunctionPO = set.CreateFunctionMemoryObject(localFunction);
  const auto localFunctionRegisterPO = set.CreateRegisterPointerObject(localFunctionRegister);
  const auto exportedFunctionPO = set.CreateFunctionMemoryObject(exportedFunction);
  set.MapRegisterToExistingPointerObject(exportedFunctionReturn, localFunctionRegisterPO);

  // Make localFunc's output point to localFunc
  set.AddToPointsToSet(localFunctionRegisterPO, localFunctionPO);

  // Make a constraints set
  PointerObjectConstraintSet constraints(set);
  constraints.AddConstraint(HandleEscapingFunctionConstraint(exportedFunctionPO));
  constraints.Solve();

  // Nothing has happened yet, since the exported function is yet to be marked as escaped
  assert(!set.GetPointerObject(localFunctionPO).HasEscaped());
  assert(!set.GetPointerObject(localFunctionPO).HasEscaped());
  assert(!set.GetPointerObject(exportedFunctionPO).HasEscaped());

  set.GetPointerObject(exportedFunctionPO).MarkAsEscaped();
  constraints.Solve();

  // Now the local function has been marked as escaped as well, since it is the return value
  assert(set.GetPointerObject(localFunctionPO).HasEscaped());
}

static void
TestAddPointsToExternalConstraint()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(2);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1));
  auto reg1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));

  PointerObjectConstraintSet constraints(set);
  constraints.AddPointerPointeeConstraint(reg0, alloca0);
  constraints.AddPointerPointeeConstraint(reg1, alloca1);

  constraints.AddPointsToExternalConstraint(reg0);
  constraints.Solve();

  // Make sure only reg0 points to external, and nothing has escaped
  assert(set.GetPointerObject(reg0).PointsToExternal());
  assert(!set.GetPointerObject(reg1).PointsToExternal());
  assert(!set.GetPointerObject(alloca0).PointsToExternal());
  assert(!set.GetPointerObject(alloca1).PointsToExternal());

  assert(!set.GetPointerObject(reg0).HasEscaped());
  assert(!set.GetPointerObject(reg1).HasEscaped());
  assert(!set.GetPointerObject(alloca0).HasEscaped());
  assert(!set.GetPointerObject(alloca1).HasEscaped());

  // Add a *reg0 = reg1 store
  constraints.AddConstraint(AllPointeesPointToSupersetConstraint(reg0, reg1));
  constraints.Solve();

  // Now alloca1 is marked as escaped, due to being written to a pointer that might point to
  // external
  assert(set.GetPointerObject(alloca1).HasEscaped());
  // The other alloca has not escaped
  assert(!set.GetPointerObject(alloca0).HasEscaped());
}

static void
TestAddRegisterContentEscapedConstraint()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(2);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1));
  auto reg1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));

  PointerObjectConstraintSet constraints(set);
  constraints.AddPointerPointeeConstraint(reg0, alloca0);
  constraints.AddPointerPointeeConstraint(reg1, alloca1);

  // return reg0, making alloca0 escapce
  constraints.AddRegisterContentEscapedConstraint(reg0);
  constraints.Solve();

  // Make sure only alloca0 has escaped
  assert(set.GetPointerObject(alloca0).HasEscaped());
  assert(!set.GetPointerObject(alloca1).HasEscaped());

  // Add a alloca0 = reg1 store
  constraints.AddConstraint(AllPointeesPointToSupersetConstraint(reg0, reg1));
  constraints.Solve();

  // Now both are marked as escaped
  assert(set.GetPointerObject(alloca0).HasEscaped());
  assert(set.GetPointerObject(alloca1).HasEscaped());
}

// Tests crating a ConstraintSet with multiple different constraints and calling Solve()
static void
TestPointerObjectConstraintSetSolve()
{
  using namespace jlm::llvm::aa;

  // Create a graph with 11 different registers, and 4 allocas.
  jlm::tests::NAllocaNodesTest rvsdg(4);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  PointerObject::Index reg[11];
  for (size_t i = 0; i < 11; i++)
    reg[i] = set.CreateDummyRegisterPointerObject();

  // %0 is a function parameter
  // %1 = alloca 8 (variable v1)
  // %2 = alloca 8 (variable v2)
  // %3 = alloca 8 (variable v3)
  // %4 = alloca 8 (variable v4)
  auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  auto alloca2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1));
  auto alloca3 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2));
  auto alloca4 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(3));

  // Now start building constraints based on instructions
  PointerObjectConstraintSet constraints(set);

  // the function parameter is marked as pointing to external
  constraints.AddPointsToExternalConstraint(reg[0]);

  // all alloca outputs point to the alloca memory object
  constraints.AddPointerPointeeConstraint(reg[1], alloca1);
  constraints.AddPointerPointeeConstraint(reg[2], alloca2);
  constraints.AddPointerPointeeConstraint(reg[3], alloca3);
  constraints.AddPointerPointeeConstraint(reg[4], alloca4);

  // store [%1], %2 (alloca1 now points to alloca2)
  constraints.AddConstraint(AllPointeesPointToSupersetConstraint(reg[1], reg[2]));
  // store [%2], %3 (alloca2 now points to alloca3)
  constraints.AddConstraint(AllPointeesPointToSupersetConstraint(reg[2], reg[3]));
  // store [%3], %4 (alloca3 now points to alloca4)
  constraints.AddConstraint(AllPointeesPointToSupersetConstraint(reg[3], reg[4]));

  // %5 = load [%1] (loads v1, which is %2, the pointer to alloca2)
  constraints.AddConstraint(SupersetOfAllPointeesConstraint(reg[5], reg[1]));
  // %6 = load [%3] (loads v3, which is %4, the pointer to alloca4)
  constraints.AddConstraint(SupersetOfAllPointeesConstraint(reg[6], reg[3]));

  // %7 = phi %5/%6 (either points to alloca2 or alloca4)
  constraints.AddConstraint(SupersetConstraint(reg[7], reg[5]));
  constraints.AddConstraint(SupersetConstraint(reg[7], reg[6]));

  // %8 = phi %0/%4 (either the function argument or pointer to alloca4)
  constraints.AddConstraint(SupersetConstraint(reg[8], reg[0]));
  constraints.AddConstraint(SupersetConstraint(reg[8], reg[4]));

  // %9 = load [%7] (either loads alloca2 or alloca4, the first is a pointer to alloca3)
  constraints.AddConstraint(SupersetOfAllPointeesConstraint(reg[9], reg[7]));

  // store [%8], %9 (stores what might be a pointer to alloca3 into what's either alloca4 or the
  // pointer argument)
  constraints.AddConstraint(AllPointeesPointToSupersetConstraint(reg[8], reg[9]));

  // %10 = load [%8] (loads from possibly external, should also point to external)
  constraints.AddConstraint(SupersetOfAllPointeesConstraint(reg[10], reg[8]));

  // Find a solution to all the constraints
  constraints.Solve();

  // alloca1 should point to alloca2, etc
  assert(set.GetPointsToSet(alloca1).Size() == 1);
  assert(set.GetPointsToSet(alloca1).Contains(alloca2));
  assert(set.GetPointsToSet(alloca2).Size() == 1);
  assert(set.GetPointsToSet(alloca2).Contains(alloca3));
  assert(set.GetPointsToSet(alloca3).Size() == 1);
  assert(set.GetPointsToSet(alloca3).Contains(alloca4));

  // %5 is a load of alloca1, and should only be a pointer to alloca2
  assert(set.GetPointsToSet(reg[5]).Size() == 1);
  assert(set.GetPointsToSet(reg[5]).Contains(alloca2));

  // %6 is a load of alloca3, and should only be a pointer to alloca4
  assert(set.GetPointsToSet(reg[6]).Size() == 1);
  assert(set.GetPointsToSet(reg[6]).Contains(alloca4));

  // %7 can point to either alloca2 or alloca4
  assert(set.GetPointsToSet(reg[7]).Size() == 2);
  assert(set.GetPointsToSet(reg[7]).Contains(alloca2));
  assert(set.GetPointsToSet(reg[7]).Contains(alloca4));

  // %8 should point to external, since it points to the superset of %0 and %1
  assert(set.GetPointerObject(reg[8]).PointsToExternal());
  // %8 may also point to alloca4
  assert(set.GetPointsToSet(reg[8]).Size() == 1);
  assert(set.GetPointsToSet(reg[8]).Contains(alloca4));

  // %9 may point to v3
  assert(set.GetPointsToSet(reg[9]).Contains(alloca3));

  // Due to the store of %9 into [%8], alloca4 may now point back to alloca3
  assert(set.GetPointsToSet(alloca4).Size() == 1);
  assert(set.GetPointsToSet(alloca4).Contains(alloca3));

  // Also due to the same store, alloca3 might have escaped
  assert(set.GetPointerObject(alloca3).HasEscaped());
  // Due to alloca3 pointing to alloca4, it too should have been marked as escaped
  assert(set.GetPointerObject(alloca4).HasEscaped());
  // Check that the other two allocas haven't escaped
  assert(!set.GetPointerObject(alloca1).HasEscaped());
  assert(!set.GetPointerObject(alloca2).HasEscaped());

  // Make sure only the escaped allocas are marked as pointing to external
  assert(!set.GetPointerObject(alloca1).PointsToExternal());
  assert(!set.GetPointerObject(alloca2).PointsToExternal());
  assert(set.GetPointerObject(alloca3).PointsToExternal());
  assert(set.GetPointerObject(alloca4).PointsToExternal());

  // %10 should also point to external, since it might have been loaded from external
  assert(set.GetPointerObject(reg[10]).PointsToExternal());
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
  TestHandleEscapingFunctionConstraint();
  TestAddPointsToExternalConstraint();
  TestAddRegisterContentEscapedConstraint();
  TestPointerObjectConstraintSetSolve();
  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestPointerObjectSet", TestPointerObjectSet)

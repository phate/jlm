/*
 * Copyright 2023, 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>
#include <jlm/llvm/TestRvsdgs.hpp>

#include <cassert>

static bool
StringContains(std::string_view haystack, std::string_view needle)
{
  return haystack.find(needle) != std::string::npos;
}

// Test the flag functions on the PointerObject class
TEST(PointerObjectSetTests, TestFlagFunctions)
{
  using namespace jlm::llvm::aa;

  jlm::llvm::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  PointerObjectSet set;
  auto registerPO = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput());

  EXPECT_TRUE(set.CanPoint(registerPO));
  EXPECT_TRUE(set.IsPointerObjectRegister(registerPO));

  // PointeesEscaping flag
  EXPECT_FALSE(set.HasPointeesEscaping(registerPO));
  EXPECT_TRUE(set.MarkAsPointeesEscaping(registerPO));
  EXPECT_TRUE(set.HasPointeesEscaping(registerPO));
  // Trying to set the flag again returns false
  EXPECT_FALSE(set.MarkAsPointeesEscaping(registerPO));
  EXPECT_TRUE(set.HasPointeesEscaping(registerPO));

  // PointsToExternal flag. For registers, the two flags are completely independent.
  EXPECT_FALSE(set.IsPointingToExternal(registerPO));
  EXPECT_TRUE(set.MarkAsPointingToExternal(registerPO));
  EXPECT_TRUE(set.IsPointingToExternal(registerPO));
  // Trying to set the flag again returns false
  EXPECT_FALSE(set.MarkAsPointingToExternal(registerPO));
  EXPECT_TRUE(set.IsPointingToExternal(registerPO));

  // Create a new PointerObject to start with empty flags
  auto allocaPO = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(), true);
  EXPECT_FALSE(set.IsPointerObjectRegister(allocaPO));

  // Escaping means another module can write a pointer to you.
  // This implies another module might override it with pointers to external.
  // It also implies any pointees should also escape
  EXPECT_FALSE(set.IsPointingToExternal(allocaPO));
  EXPECT_FALSE(set.HasPointeesEscaping(allocaPO));
  EXPECT_TRUE(set.MarkAsEscaped(allocaPO));
  EXPECT_TRUE(set.IsPointingToExternal(allocaPO));
  EXPECT_TRUE(set.HasPointeesEscaping(allocaPO));
  // Already marked with these flags, trying to set them again makes no difference
  EXPECT_FALSE(set.MarkAsPointingToExternal(allocaPO));
  EXPECT_FALSE(set.MarkAsPointeesEscaping(allocaPO));
}

// Test creating pointer objects for each type of memory node
TEST(PointerObjectSetTests, TestCreatePointerObjects)
{
  using namespace jlm::llvm::aa;

  jlm::llvm::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  PointerObjectSet set;
  // Register PointerObjects have some extra ways of being created: Dummy and mapping
  const auto register0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput());
  set.MapRegisterToExistingPointerObject(rvsdg.GetDeltaOutput(), register0);
  const auto dummy0 = set.CreateDummyRegisterPointerObject();

  // For PointerObjects representing MemoryObjects, there is only one Create function
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(), false);
  const auto malloc0 = set.CreateMallocMemoryObject(rvsdg.GetMallocNode(), true);
  const auto delta0 = set.CreateGlobalMemoryObject(rvsdg.GetDeltaNode(), true);
  const auto lambda0 = set.CreateFunctionMemoryObject(rvsdg.GetLambdaNode());
  const auto import0 = set.CreateImportMemoryObject(rvsdg.GetImportOutput(), false);

  EXPECT_EQ(set.NumPointerObjects(), 7);
  EXPECT_EQ(set.NumPointerObjectsOfKind(jlm::llvm::aa::PointerObjectKind::Register), 2);

  EXPECT_EQ(set.GetPointerObjectKind(register0), PointerObjectKind::Register);
  EXPECT_EQ(set.GetPointerObjectKind(dummy0), PointerObjectKind::Register);
  EXPECT_EQ(set.GetPointerObjectKind(alloca0), PointerObjectKind::AllocaMemoryObject);
  EXPECT_EQ(set.GetPointerObjectKind(malloc0), PointerObjectKind::MallocMemoryObject);
  EXPECT_EQ(set.GetPointerObjectKind(delta0), PointerObjectKind::GlobalMemoryObject);
  EXPECT_EQ(set.GetPointerObjectKind(lambda0), PointerObjectKind::FunctionMemoryObject);
  EXPECT_EQ(set.GetPointerObjectKind(import0), PointerObjectKind::ImportMemoryObject);

  // Most pointer objects don't start out as escaped
  EXPECT_FALSE(set.HasEscaped(dummy0));
  EXPECT_FALSE(set.HasEscaped(alloca0));
  EXPECT_FALSE(set.HasEscaped(malloc0));
  EXPECT_FALSE(set.HasEscaped(delta0));
  EXPECT_FALSE(set.HasEscaped(lambda0));
  // ...but imported objects are always escaped
  EXPECT_TRUE(set.HasEscaped(import0));
  // ...which also means it points to external, and has its pointees escaping
  EXPECT_TRUE(set.IsPointingToExternal(import0) && set.HasPointeesEscaping(import0));

  // Some kinds of PointerObjects have CanPoint() configurable is the constructor
  EXPECT_FALSE(set.CanPoint(alloca0));
  EXPECT_TRUE(set.CanPoint(malloc0));
  EXPECT_TRUE(set.CanPoint(delta0));
  // ...while others have implied values of CanPoint()
  EXPECT_TRUE(set.CanPoint(register0));
  EXPECT_FALSE(set.CanPoint(lambda0));
  EXPECT_FALSE(set.CanPoint(import0));

  // CanPoint() == false implies pointing to external and having all pointees escaping
  EXPECT_TRUE(set.IsPointingToExternal(alloca0) && set.HasPointeesEscaping(alloca0));

  // Registers have helper function for looking up existing PointerObjects
  EXPECT_EQ(set.GetRegisterPointerObject(rvsdg.GetAllocaOutput()), register0);
  EXPECT_EQ(set.GetRegisterPointerObject(rvsdg.GetDeltaOutput()), register0);
  EXPECT_EQ(set.TryGetRegisterPointerObject(rvsdg.GetDeltaOutput()).value(), register0);

  // Functions have the same, but also in the other direction
  EXPECT_EQ(set.GetFunctionMemoryObject(rvsdg.GetLambdaNode()), lambda0);
  EXPECT_EQ(&set.GetLambdaNodeFromFunctionMemoryObject(lambda0), &rvsdg.GetLambdaNode());

  // The maps can also be accessed directly
  EXPECT_EQ(set.GetRegisterMap().at(&rvsdg.GetAllocaOutput()), register0);
  EXPECT_EQ(set.GetRegisterMap().at(&rvsdg.GetDeltaOutput()), register0);
  EXPECT_EQ(set.GetAllocaMap().at(&rvsdg.GetAllocaNode()), alloca0);
  EXPECT_EQ(set.GetMallocMap().at(&rvsdg.GetMallocNode()), malloc0);
  EXPECT_EQ(set.GetGlobalMap().at(&rvsdg.GetDeltaNode()), delta0);
  EXPECT_EQ(set.GetFunctionMap().LookupKey(&rvsdg.GetLambdaNode()), lambda0);
  EXPECT_EQ(set.GetImportMap().at(&rvsdg.GetImportOutput()), import0);
}

TEST(PointerObjectSetTests, TestPointerObjectUnification)
{
  using namespace jlm::llvm::aa;

  PointerObjectSet set;
  auto dummy0 = set.CreateDummyRegisterPointerObject();
  auto dummy1 = set.CreateDummyRegisterPointerObject();
  EXPECT_TRUE(set.IsUnificationRoot(dummy0));

  auto root = set.UnifyPointerObjects(dummy0, dummy1);
  EXPECT_EQ(set.GetUnificationRoot(dummy0), root);
  EXPECT_EQ(set.GetUnificationRoot(dummy1), root);

  // Exactly one of the PointerObjects is the GetRootRegion
  EXPECT_NE((root == dummy0), (root == dummy1));
  EXPECT_TRUE(set.IsUnificationRoot(root));

  // Trying to unify again gives the same GetRootRegion
  EXPECT_EQ(set.UnifyPointerObjects(dummy0, dummy1), root);

  auto notRoot = dummy0 + dummy1 - root;
  EXPECT_FALSE(set.IsUnificationRoot(notRoot));

  auto dummy2 = set.CreateDummyRegisterPointerObject();
  auto dummy3 = set.CreateDummyRegisterPointerObject();
  set.UnifyPointerObjects(dummy0, dummy2);
  auto newRoot = set.UnifyPointerObjects(dummy1, dummy3);
  EXPECT_EQ(set.GetUnificationRoot(dummy0), newRoot);
  EXPECT_EQ(set.GetUnificationRoot(dummy1), newRoot);
  EXPECT_EQ(set.GetUnificationRoot(dummy2), newRoot);
  EXPECT_EQ(set.GetUnificationRoot(dummy3), newRoot);
}

TEST(PointerObjectSetTests, TestPointerObjectUnificationPointees)
{
  using namespace jlm::llvm::aa;

  jlm::llvm::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  PointerObjectSet set;
  auto lambda0 = set.CreateFunctionMemoryObject(rvsdg.GetLambdaNode());
  auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(), true);
  auto delta0 = set.CreateGlobalMemoryObject(rvsdg.GetDeltaNode(), true);

  set.AddToPointsToSet(alloca0, lambda0);
  EXPECT_EQ(set.GetPointsToSet(alloca0).Size(), 1);
  EXPECT_EQ(set.GetPointsToSet(delta0).Size(), 0);

  set.UnifyPointerObjects(delta0, alloca0);

  // Now both should share points-to sets
  EXPECT_TRUE(set.GetPointsToSet(alloca0).Contains(lambda0));
  EXPECT_TRUE(set.GetPointsToSet(delta0).Contains(lambda0));

  // Marking one as pointing to external marks all as pointing to external
  EXPECT_TRUE(set.MarkAsPointingToExternal(alloca0));
  EXPECT_TRUE(set.IsPointingToExternal(alloca0));
  EXPECT_TRUE(set.IsPointingToExternal(delta0));

  // Marking one as pointees escaping marks all as pointees escaping
  EXPECT_TRUE(set.MarkAsPointeesEscaping(delta0));
  EXPECT_TRUE(set.HasPointeesEscaping(alloca0));
  EXPECT_TRUE(set.HasPointeesEscaping(delta0));

  // Adding a new pointee adds it to all members
  auto import0 = set.CreateImportMemoryObject(rvsdg.GetImportOutput(), false);
  EXPECT_TRUE(set.AddToPointsToSet(delta0, import0));
  EXPECT_TRUE(set.GetPointsToSet(alloca0).Contains(import0));

  // Escaping is not shared within the unification
  EXPECT_TRUE(set.MarkAsEscaped(delta0));
  EXPECT_TRUE(set.HasEscaped(delta0));
  EXPECT_FALSE(set.HasEscaped(alloca0));
}

// Test the PointerObjectSet method for adding pointer objects to another pointer object's
// points-to-set
TEST(PointerObjectSetTests, TestAddToPointsToSet)
{
  using namespace jlm::llvm::aa;

  jlm::llvm::NAllocaNodesTest rvsdg(1);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0), false);
  const auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));

  EXPECT_EQ(set.GetPointsToSet(reg0).Size(), 0);

  EXPECT_TRUE(set.AddToPointsToSet(reg0, alloca0));
  EXPECT_EQ(set.GetPointsToSet(reg0).Size(), 1);
  EXPECT_TRUE(set.GetPointsToSet(reg0).Contains(alloca0));

  // Trying to add it again returns false
  EXPECT_FALSE(set.AddToPointsToSet(reg0, alloca0));
}

// Test the PointerObjectSet method for making one points-to-set a superset of another
TEST(PointerObjectSetTests, TestMakePointsToSetSuperset)
{
  using namespace jlm::llvm::aa;

  jlm::llvm::NAllocaNodesTest rvsdg(3);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0), false);
  const auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  const auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1), false);
  const auto reg1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));
  const auto alloca2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2), false);

  set.AddToPointsToSet(reg0, alloca0);
  set.AddToPointsToSet(reg1, alloca1);

  EXPECT_EQ(set.GetPointsToSet(reg0).Size(), 1);
  EXPECT_TRUE(set.GetPointsToSet(reg0).Contains(alloca0));

  EXPECT_TRUE(set.MakePointsToSetSuperset(reg0, reg1));
  EXPECT_EQ(set.GetPointsToSet(reg1).Size(), 1);
  EXPECT_EQ(set.GetPointsToSet(reg0).Size(), 2);
  EXPECT_TRUE(set.GetPointsToSet(reg0).Contains(alloca1));

  // Calling it again without changing reg1 makes no difference, returns false
  EXPECT_FALSE(set.MakePointsToSetSuperset(reg0, reg1));

  // Add an additional member to P(reg1)
  set.AddToPointsToSet(reg1, alloca2);
  EXPECT_TRUE(set.MakePointsToSetSuperset(reg0, reg1));
  EXPECT_TRUE(set.GetPointsToSet(reg0).Contains(alloca2));
}

TEST(PointerObjectSetTests, TestClonePointerObjectSet)
{
  using namespace jlm::llvm::aa;
  jlm::llvm::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto register0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput());
  const auto dummy0 = set.CreateDummyRegisterPointerObject();
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(), false);
  const auto malloc0 = set.CreateMallocMemoryObject(rvsdg.GetMallocNode(), true);
  const auto delta0 = set.CreateGlobalMemoryObject(rvsdg.GetDeltaNode(), false);
  const auto lambda0 = set.CreateFunctionMemoryObject(rvsdg.GetLambdaNode());
  const auto import0 = set.CreateImportMemoryObject(rvsdg.GetImportOutput(), false);

  set.AddToPointsToSet(register0, alloca0);
  set.UnifyPointerObjects(delta0, import0);

  auto clonedSet = set.Clone();

  // All mappings are identical, since PointerObjects are referenced by index
  EXPECT_EQ(clonedSet->NumPointerObjects(), set.NumPointerObjects());
  EXPECT_EQ(clonedSet->GetRegisterMap(), set.GetRegisterMap());
  EXPECT_EQ(clonedSet->GetAllocaMap(), set.GetAllocaMap());
  EXPECT_EQ(clonedSet->GetMallocMap(), set.GetMallocMap());
  EXPECT_EQ(clonedSet->GetGlobalMap(), set.GetGlobalMap());
  EXPECT_EQ(clonedSet->GetFunctionMap(), set.GetFunctionMap());
  EXPECT_EQ(clonedSet->GetImportMap(), set.GetImportMap());

  // In the cloned set, each points-to set should be identical
  EXPECT_EQ(clonedSet->GetPointsToSet(register0), set.GetPointsToSet(register0));

  // Unifications are maintained
  EXPECT_EQ(clonedSet->GetUnificationRoot(delta0), clonedSet->GetUnificationRoot(import0));

  // Additional changes only affect the set they are applied to
  set.AddToPointsToSet(register0, lambda0);
  EXPECT_FALSE(clonedSet->GetPointsToSet(register0).Contains(lambda0));

  clonedSet->UnifyPointerObjects(delta0, dummy0);
  EXPECT_NE(set.GetUnificationRoot(delta0), set.GetUnificationRoot(dummy0));

  set.MarkAsPointingToExternal(malloc0);
  EXPECT_FALSE(clonedSet->IsPointingToExternal(malloc0));
}

// Test the SupersetConstraint's Apply function
TEST(PointerObjectSetTests, TestSupersetConstraint)
{
  using namespace jlm::llvm::aa;

  jlm::llvm::NAllocaNodesTest rvsdg(3);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0), true);
  const auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  const auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1), true);
  const auto reg1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));
  const auto alloca2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2), true);
  const auto reg2 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(2));

  set.AddToPointsToSet(reg0, alloca0);
  set.AddToPointsToSet(reg1, alloca1);
  set.AddToPointsToSet(reg2, alloca2);

  // Make alloca0 point to everything reg1 points to
  SupersetConstraint c1(alloca0, reg1);
  EXPECT_TRUE(c1.ApplyDirectly(set));
  while (c1.ApplyDirectly(set))
    ;
  // For now this only makes alloca0 point to alloca1
  EXPECT_EQ(set.GetPointsToSet(alloca0).Size(), 1);
  EXPECT_TRUE(set.GetPointsToSet(alloca0).Contains(alloca1));

  // Make reg1 point to everything reg2 points to
  SupersetConstraint c2(reg1, reg2);
  EXPECT_TRUE(c2.ApplyDirectly(set));
  while (c2.ApplyDirectly(set))
    ;
  // This makes alloca2 a member of P(reg1)
  EXPECT_TRUE(set.GetPointsToSet(reg1).Contains(alloca2));

  // Apply c1 again
  EXPECT_TRUE(c1.ApplyDirectly(set));
  while (c1.ApplyDirectly(set))
    ;
  // Now alloca0 should also point to alloca2
  EXPECT_EQ(set.GetPointsToSet(alloca0).Size(), 2);
  EXPECT_TRUE(set.GetPointsToSet(alloca0).Contains(alloca2));

  // Make reg2 point to external, and propagate through constraints
  set.MarkAsPointingToExternal(reg2);
  EXPECT_TRUE(c2.ApplyDirectly(set));
  while (c2.ApplyDirectly(set))
    ;
  EXPECT_TRUE(c1.ApplyDirectly(set));
  while (c1.ApplyDirectly(set))
    ;
  // Now both reg1 and alloca0 may point to external
  EXPECT_TRUE(set.IsPointingToExternal(reg1));
  EXPECT_TRUE(set.IsPointingToExternal(alloca0));
}

TEST(PointerObjectSetTests, TestStoreConstraintDirectly)
{
  using namespace jlm::llvm::aa;

  jlm::llvm::NAllocaNodesTest rvsdg(3);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0), true);
  const auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  const auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1), true);
  const auto reg1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));
  const auto alloca2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2), true);
  const auto reg2 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(2));

  set.AddToPointsToSet(reg0, alloca0);
  set.AddToPointsToSet(reg1, alloca1);
  set.AddToPointsToSet(reg2, alloca2);

  // Add *alloca0 = reg2, which will do nothing, since alloca0 can't be pointing to anything yet
  StoreConstraint c1(alloca0, reg2);
  EXPECT_FALSE(c1.ApplyDirectly(set));

  // This means *reg0 = reg1, and as we know, reg0 points to alloca0
  // This should make alloca0 point to anything reg1 points to, aka alloca1
  StoreConstraint c2(reg0, reg1);
  EXPECT_TRUE(c2.ApplyDirectly(set));
  while (c2.ApplyDirectly(set))
    ;
  EXPECT_EQ(set.GetPointsToSet(alloca0).Size(), 1);
  EXPECT_TRUE(set.GetPointsToSet(alloca0).Contains(alloca1));

  // Do c1 again, now that alloca0 points to alloca1
  EXPECT_TRUE(c1.ApplyDirectly(set));
  while (c1.ApplyDirectly(set))
    ;
  EXPECT_EQ(set.GetPointsToSet(alloca1).Size(), 1);
  EXPECT_TRUE(set.GetPointsToSet(alloca1).Contains(alloca2));
}

TEST(PointerObjectSetTests, TestLoadConstraintDirectly)
{
  using namespace jlm::llvm::aa;

  jlm::llvm::NAllocaNodesTest rvsdg(3);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0), true);
  const auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  const auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1), true);
  const auto reg1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));
  const auto alloca2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2), true);
  const auto reg2 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(2));

  set.AddToPointsToSet(reg0, alloca0);
  set.AddToPointsToSet(reg1, alloca1);
  set.AddToPointsToSet(reg2, alloca2);

  // Makes reg1 = *reg0, where reg0 currently just points to alloca0
  // Since alloca0 currently has no pointees, this does nothing
  LoadConstraint c1(reg1, reg0);
  EXPECT_FALSE(c1.ApplyDirectly(set));

  // Make alloca0 point to something: alloca2
  set.AddToPointsToSet(alloca0, alloca2);
  // The constraint now makes a difference
  EXPECT_TRUE(c1.ApplyDirectly(set));
  while (c1.ApplyDirectly(set))
    ;
  EXPECT_EQ(set.GetPointsToSet(reg1).Size(), 2);
  EXPECT_TRUE(set.GetPointsToSet(reg1).Contains(alloca2));
}

TEST(PointerObjectSetTests, TestEscapedFunctionConstraint)
{
  using namespace jlm::llvm::aa;

  jlm::llvm::EscapingLocalFunctionTest rvsdg;
  rvsdg.InitializeTest();

  const auto & localFunction = rvsdg.GetLocalFunction();
  const auto & localFunctionRegister = rvsdg.GetLocalFunctionRegister();
  const auto & exportedFunction = rvsdg.GetExportedFunction();
  const auto & exportedFunctionReturn = *exportedFunction.GetFunctionResults()[0]->origin();

  PointerObjectSet set;
  const auto localFunctionPO = set.CreateFunctionMemoryObject(localFunction);
  const auto localFunctionRegisterPO = set.CreateRegisterPointerObject(localFunctionRegister);
  const auto exportedFunctionPO = set.CreateFunctionMemoryObject(exportedFunction);
  set.MapRegisterToExistingPointerObject(exportedFunctionReturn, localFunctionRegisterPO);

  // Make localFunc's output point to localFunc
  set.AddToPointsToSet(localFunctionRegisterPO, localFunctionPO);

  bool result = EscapedFunctionConstraint::PropagateEscapedFunctionsDirectly(set);

  // Nothing has happened yet, since the exported function is yet to be marked as escaped
  EXPECT_FALSE(result);
  EXPECT_FALSE(set.HasEscaped(localFunctionRegisterPO));
  EXPECT_FALSE(set.HasEscaped(exportedFunctionPO));

  set.MarkAsEscaped(exportedFunctionPO);

  // Use both EscapedFunctionConstraint and EscapeFlagConstraint to propagate flags
  result = EscapedFunctionConstraint::PropagateEscapedFunctionsDirectly(set);
  EXPECT_TRUE(result);

  // Now the return value is marked as all pointees escaping, so make that happen
  result = EscapeFlagConstraint::PropagateEscapedFlagsDirectly(set);
  // Now the local function has been marked as escaped as well, since it is the return value
  EXPECT_TRUE(result);
  EXPECT_TRUE(set.HasEscaped(localFunctionPO));
}

TEST(PointerObjectSetTests, TestStoredAsScalarFlag)
{
  using namespace jlm::llvm::aa;

  jlm::llvm::NAllocaNodesTest rvsdg(3);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto p0 = set.CreateDummyRegisterPointerObject();
  const auto p1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0), true);
  const auto p11 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1), true);
  const auto p2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2), true);

  set.AddToPointsToSet(p0, p1);
  set.AddToPointsToSet(p1, p11);
  set.AddToPointsToSet(p0, p2);

  bool result = EscapeFlagConstraint::PropagateEscapedFlagsDirectly(set);
  EXPECT_FALSE(result);

  set.MarkAsStoringAsScalar(p0);
  result = EscapeFlagConstraint::PropagateEscapedFlagsDirectly(set);
  EXPECT_TRUE(result);

  // p0 should only have the single stored as scalar flag
  EXPECT_FALSE(set.HasEscaped(p0));
  EXPECT_FALSE(set.HasPointeesEscaping(p0));
  EXPECT_FALSE(set.IsPointingToExternal(p0));
  EXPECT_FALSE(set.IsLoadedAsScalar(p0));
  EXPECT_TRUE(set.IsStoredAsScalar(p0));

  // p1 and p2 should both point to external, but not any other flags
  EXPECT_FALSE(set.HasEscaped(p1));
  EXPECT_FALSE(set.HasPointeesEscaping(p1));
  EXPECT_TRUE(set.IsPointingToExternal(p1));
  EXPECT_FALSE(set.IsLoadedAsScalar(p1));
  EXPECT_FALSE(set.IsStoredAsScalar(p1));

  EXPECT_FALSE(set.HasEscaped(p2));
  EXPECT_FALSE(set.HasPointeesEscaping(p2));
  EXPECT_TRUE(set.IsPointingToExternal(p2));
  EXPECT_FALSE(set.IsLoadedAsScalar(p2));
  EXPECT_FALSE(set.IsStoredAsScalar(p2));

  // p11 should have no flags
  EXPECT_FALSE(set.HasEscaped(p11));
  EXPECT_FALSE(set.HasPointeesEscaping(p11));
  EXPECT_FALSE(set.IsPointingToExternal(p11));
  EXPECT_FALSE(set.IsLoadedAsScalar(p11));
  EXPECT_FALSE(set.IsStoredAsScalar(p11));

  // Applying again does nothing
  result = EscapeFlagConstraint::PropagateEscapedFlagsDirectly(set);
  EXPECT_FALSE(result);
}

TEST(PointerObjectSetTests, TestLoadedAsScalarFlag)
{
  using namespace jlm::llvm::aa;

  jlm::llvm::NAllocaNodesTest rvsdg(5);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto p0 = set.CreateDummyRegisterPointerObject();
  const auto p1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0), true);
  const auto p11 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1), true);
  const auto p12 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2), true);
  const auto p2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(3), true);
  const auto p21 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(4), true);

  set.AddToPointsToSet(p0, p1);
  set.AddToPointsToSet(p1, p11);
  set.AddToPointsToSet(p1, p12);
  set.AddToPointsToSet(p0, p2);
  set.AddToPointsToSet(p2, p21);

  bool result = EscapeFlagConstraint::PropagateEscapedFlagsDirectly(set);
  EXPECT_FALSE(result);

  set.MarkAsLoadingAsScalar(p0);
  result = EscapeFlagConstraint::PropagateEscapedFlagsDirectly(set);
  EXPECT_TRUE(result);

  // p0 should only have the single loaded as scalar flag
  EXPECT_FALSE(set.HasEscaped(p0));
  EXPECT_FALSE(set.HasPointeesEscaping(p0));
  EXPECT_FALSE(set.IsPointingToExternal(p0));
  EXPECT_FALSE(set.IsStoredAsScalar(p0));
  EXPECT_TRUE(set.IsLoadedAsScalar(p0));

  // p1 should only have the pointees escape flag
  EXPECT_FALSE(set.HasEscaped(p1));
  EXPECT_TRUE(set.HasPointeesEscaping(p1));
  EXPECT_FALSE(set.IsPointingToExternal(p1));
  EXPECT_FALSE(set.IsStoredAsScalar(p1));
  EXPECT_FALSE(set.IsLoadedAsScalar(p1));

  // p11, p12, p21 should have escaped, but not be flagged using the store or load flags
  EXPECT_TRUE(set.HasEscaped(p11));
  EXPECT_FALSE(set.IsLoadedAsScalar(p11));
  EXPECT_FALSE(set.IsStoredAsScalar(p11));

  EXPECT_TRUE(set.HasEscaped(p12));
  EXPECT_FALSE(set.IsLoadedAsScalar(p12));
  EXPECT_FALSE(set.IsStoredAsScalar(p12));

  EXPECT_TRUE(set.HasEscaped(p21));
  EXPECT_FALSE(set.IsLoadedAsScalar(p21));
  EXPECT_FALSE(set.IsStoredAsScalar(p21));

  // Applying again does nothing
  result = EscapeFlagConstraint::PropagateEscapedFlagsDirectly(set);
  EXPECT_FALSE(result);
}

TEST(PointerObjectSetTests, TestFunctionCallConstraint)
{
  using namespace jlm::llvm::aa;

  jlm::llvm::CallTest1 rvsdg;
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto lambdaF = set.CreateFunctionMemoryObject(*rvsdg.lambda_f);
  const auto lambdaFRegister = set.CreateRegisterPointerObject(*rvsdg.lambda_f->output());
  const auto lambdaFArgumentX =
      set.CreateRegisterPointerObject(*rvsdg.lambda_f->GetFunctionArguments()[0]);
  const auto lambdaFArgumentY =
      set.CreateRegisterPointerObject(*rvsdg.lambda_f->GetFunctionArguments()[1]);
  const auto allocaX = set.CreateAllocaMemoryObject(*rvsdg.alloca_x, true);
  const auto allocaY = set.CreateAllocaMemoryObject(*rvsdg.alloca_y, true);
  const auto allocaXRegister = set.CreateRegisterPointerObject(*rvsdg.alloca_x->output(0));
  const auto allocaYRegister = set.CreateRegisterPointerObject(*rvsdg.alloca_y->output(0));

  // Associate the allocas and lambda with their output
  set.AddToPointsToSet(lambdaFRegister, lambdaF);
  set.AddToPointsToSet(allocaXRegister, allocaX);
  set.AddToPointsToSet(allocaYRegister, allocaY);

  FunctionCallConstraint c(lambdaFRegister, rvsdg.CallF());
  EXPECT_TRUE(c.ApplyDirectly(set));
  while (c.ApplyDirectly(set))
    ;

  // The arguments to f should now be pointing to exactly the corresponding allocas
  EXPECT_TRUE(set.GetPointsToSet(lambdaFArgumentX).Contains(allocaX));
  EXPECT_TRUE(set.GetPointsToSet(lambdaFArgumentY).Contains(allocaY));
  EXPECT_FALSE(set.GetPointsToSet(lambdaFArgumentX).Contains(allocaY));
  EXPECT_FALSE(set.GetPointsToSet(lambdaFArgumentY).Contains(allocaX));
}

TEST(PointerObjectSetTests, TestAddPointsToExternalConstraint)
{
  using namespace jlm::llvm::aa;

  jlm::llvm::NAllocaNodesTest rvsdg(2);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0), true);
  const auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  const auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1), true);
  const auto reg1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));

  PointerObjectConstraintSet constraints(set);
  constraints.AddPointerPointeeConstraint(reg0, alloca0);
  constraints.AddPointerPointeeConstraint(reg1, alloca1);

  constraints.AddPointsToExternalConstraint(reg0);
  constraints.SolveNaively();

  // Make sure only reg0 points to external, and nothing has escaped
  EXPECT_TRUE(set.IsPointingToExternal(reg0));
  EXPECT_FALSE(set.IsPointingToExternal(reg1));
  EXPECT_FALSE(set.IsPointingToExternal(alloca0));
  EXPECT_FALSE(set.IsPointingToExternal(alloca1));

  EXPECT_FALSE(set.HasEscaped(reg0));
  EXPECT_FALSE(set.HasEscaped(reg1));
  EXPECT_FALSE(set.HasEscaped(alloca0));
  EXPECT_FALSE(set.HasEscaped(alloca1));

  // Add a *reg0 = reg1 store
  constraints.AddConstraint(StoreConstraint(reg0, reg1));
  constraints.SolveNaively();

  // Now alloca1 is marked as escaped, due to being written to a pointer that might point to
  // external
  EXPECT_TRUE(set.HasEscaped(alloca1));
  // The other alloca has not escaped
  EXPECT_FALSE(set.HasEscaped(alloca0));
}

TEST(PointerObjectSetTests, TestAddRegisterContentEscapedConstraint)
{
  using namespace jlm::llvm::aa;

  jlm::llvm::NAllocaNodesTest rvsdg(2);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0), false);
  const auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  const auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1), false);
  const auto reg1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));

  PointerObjectConstraintSet constraints(set);
  constraints.AddPointerPointeeConstraint(reg0, alloca0);
  constraints.AddPointerPointeeConstraint(reg1, alloca1);

  // return reg0, making alloca0 escapce
  constraints.AddRegisterContentEscapedConstraint(reg0);
  constraints.SolveNaively();

  // Make sure only alloca0 has escaped
  EXPECT_TRUE(set.HasEscaped(alloca0));
  EXPECT_FALSE(set.HasEscaped(alloca1));

  // Add a alloca0 = reg1 store
  constraints.AddConstraint(StoreConstraint(reg0, reg1));
  constraints.SolveNaively();

  // Now both are marked as escaped
  EXPECT_TRUE(set.HasEscaped(alloca0));
  EXPECT_TRUE(set.HasEscaped(alloca1));
}

TEST(PointerObjectSetTests, TestDrawSubsetGraph)
{
  using namespace jlm::llvm::aa;
  using namespace jlm::util;
  jlm::llvm::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  // Arrange
  PointerObjectSet set;
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(), true);
  const auto allocaReg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput());

  const auto dummy0 = set.CreateDummyRegisterPointerObject();
  const auto dummy1 = set.CreateDummyRegisterPointerObject();
  const auto root = set.UnifyPointerObjects(dummy0, dummy1);
  const auto nonRoot = dummy0 + dummy1 - root;

  const auto storeValue = set.CreateDummyRegisterPointerObject();
  const auto storePointer = set.CreateDummyRegisterPointerObject();

  const auto loadValue = set.CreateDummyRegisterPointerObject();
  const auto loadPointer = set.CreateDummyRegisterPointerObject();

  const auto function0 = set.CreateFunctionMemoryObject(rvsdg.GetLambdaNode());

  const auto import0 = set.CreateImportMemoryObject(rvsdg.GetImportOutput(), false);

  PointerObjectConstraintSet constraints(set);
  constraints.AddPointsToExternalConstraint(nonRoot);
  constraints.AddPointerPointeeConstraint(allocaReg0, alloca0);
  constraints.AddConstraint(SupersetConstraint(import0, allocaReg0));
  constraints.AddConstraint(StoreConstraint(storePointer, storeValue));
  constraints.AddConstraint(LoadConstraint(loadValue, loadPointer));

  // Act
  graph::Writer writer;
  auto & graph = constraints.DrawSubsetGraph(writer);

  // Assert
  EXPECT_EQ(graph.NumNodes(), set.NumPointerObjects());

  // Check that the unified node that is not the GetRootRegion, contains the index of the
  // GetRootRegion
  EXPECT_TRUE(StringContains(graph.GetNode(nonRoot).GetLabel(), "#" + std::to_string(root)));

  // Check that the unification GetRootRegion's label indicates pointing to external
  EXPECT_TRUE(StringContains(graph.GetNode(root).GetLabel(), "{+}"));

  // Check that allocaReg0 points to alloca0
  EXPECT_TRUE(StringContains(graph.GetNode(allocaReg0).GetLabel(), strfmt("{", alloca0, "}")));

  // Check that a regular edge connects allocaReg0 to the importNode
  auto * supersetEdge = graph.GetEdgeBetween(graph.GetNode(allocaReg0), graph.GetNode(import0));
  EXPECT_TRUE(supersetEdge);
  EXPECT_TRUE(supersetEdge->IsDirected());
  EXPECT_EQ(supersetEdge->GetAttributeString("style").value_or("solid"), "solid");

  // Check that a store edge connects storeValue to storePointer
  auto * storeEdge = graph.GetEdgeBetween(graph.GetNode(storeValue), graph.GetNode(storePointer));
  EXPECT_TRUE(storeEdge);
  EXPECT_TRUE(storeEdge->IsDirected());
  EXPECT_TRUE(storeEdge->GetAttributeString("style") == graph::Edge::Style::Dashed);
  EXPECT_TRUE(StringContains(storeEdge->GetAttributeString("arrowhead").value(), "dot"));

  // Check that a load edge connects loadPointer to loadValue
  auto * loadEdge = graph.GetEdgeBetween(graph.GetNode(loadPointer), graph.GetNode(loadValue));
  EXPECT_TRUE(loadEdge);
  EXPECT_TRUE(loadEdge->IsDirected());
  EXPECT_TRUE(loadEdge->GetAttributeString("style") == graph::Edge::Style::Dashed);
  EXPECT_TRUE(StringContains(loadEdge->GetAttributeString("arrowtail").value(), "dot"));

  // Check that the function contains the word "function0"
  auto & functionNode = graph.GetNode(function0);
  EXPECT_TRUE(StringContains(functionNode.GetLabel(), "function0"));
  // Since functions don't track pointees, they should have CantPoint
  EXPECT_TRUE(StringContains(functionNode.GetLabel(), "CantPoint"));
  // They should also both point to external, and escape all pointees
  EXPECT_TRUE(StringContains(functionNode.GetLabel(), "{+}e"));

  // Check that the import PointerObject has a fill color, since it has escaped
  auto & importNode = graph.GetNode(import0);
  EXPECT_TRUE(importNode.HasAttribute("fillcolor"));
}

// Tests crating a ConstraintSet with multiple different constraints and calling Solve()
template<jlm::llvm::aa::Andersen::Configuration::Solver solver, typename... Args>
static void
TestPointerObjectConstraintSetSolve(Args... args)
{
  using namespace jlm::llvm::aa;

  // Create a graph with 11 different registers, and 4 allocas.
  jlm::llvm::NAllocaNodesTest rvsdg(4);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  PointerObjectIndex reg[11];
  for (unsigned int & i : reg)
    i = set.CreateDummyRegisterPointerObject();

  // %0 is a function parameter
  // %1 = alloca 8 (variable v1)
  // %2 = alloca 8 (variable v2)
  // %3 = alloca 8 (variable v3)
  // %4 = alloca 8 (variable v4)
  const auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0), true);
  const auto alloca2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1), true);
  const auto alloca3 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2), true);
  const auto alloca4 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(3), true);

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
  constraints.AddConstraint(StoreConstraint(reg[1], reg[2]));
  // store [%2], %3 (alloca2 now points to alloca3)
  constraints.AddConstraint(StoreConstraint(reg[2], reg[3]));
  // store [%3], %4 (alloca3 now points to alloca4)
  constraints.AddConstraint(StoreConstraint(reg[3], reg[4]));

  // %5 = load [%1] (loads v1, which is %2, the pointer to alloca2)
  constraints.AddConstraint(LoadConstraint(reg[5], reg[1]));
  // %6 = load [%3] (loads v3, which is %4, the pointer to alloca4)
  constraints.AddConstraint(LoadConstraint(reg[6], reg[3]));

  // %7 = phi %5/%6 (either points to alloca2 or alloca4)
  constraints.AddConstraint(SupersetConstraint(reg[7], reg[5]));
  constraints.AddConstraint(SupersetConstraint(reg[7], reg[6]));

  // %8 = phi %0/%4 (either the function argument or pointer to alloca4)
  constraints.AddConstraint(SupersetConstraint(reg[8], reg[0]));
  constraints.AddConstraint(SupersetConstraint(reg[8], reg[4]));

  // %9 = load [%7] (either loads alloca2 or alloca4, the first is a pointer to alloca3)
  constraints.AddConstraint(LoadConstraint(reg[9], reg[7]));

  // store [%8], %9 (stores what might be a pointer to alloca3 into what's either alloca4 or the
  // pointer argument)
  constraints.AddConstraint(StoreConstraint(reg[8], reg[9]));

  // %10 = load [%8] (loads from possibly external, should also point to external)
  constraints.AddConstraint(LoadConstraint(reg[10], reg[8]));

  // Find a solution to all the constraints
  if constexpr (solver == Andersen::Configuration::Solver::Worklist)
  {
    constraints.SolveUsingWorklist(args...);
  }
  else if constexpr (solver == Andersen::Configuration::Solver::Naive)
  {
    static_assert(sizeof...(args) == 0, "The naive solver takes no arguments");
    constraints.SolveNaively();
  }
  else
  {
    JLM_UNREACHABLE("Unknown solver");
  }

  // alloca1 should point to alloca2, etc
  EXPECT_LE(set.GetPointsToSet(alloca1).Size(), 1);
  EXPECT_TRUE(set.IsPointingTo(alloca1, alloca2));
  EXPECT_LE(set.GetPointsToSet(alloca2).Size(), 1);
  EXPECT_TRUE(set.IsPointingTo(alloca2, alloca3));
  EXPECT_LE(set.GetPointsToSet(alloca3).Size(), 1);
  EXPECT_TRUE(set.IsPointingTo(alloca3, alloca4));

  // %5 is a load of alloca1, and should only be a pointer to alloca2
  EXPECT_LE(set.GetPointsToSet(reg[5]).Size(), 1);
  EXPECT_TRUE(set.IsPointingTo(reg[5], alloca2));

  // %6 is a load of alloca3, and should only be a pointer to alloca4
  EXPECT_LE(set.GetPointsToSet(reg[6]).Size(), 1);
  EXPECT_TRUE(set.IsPointingTo(reg[6], alloca4));

  // %7 can point to either alloca2 or alloca4
  EXPECT_LE(set.GetPointsToSet(reg[7]).Size(), 2);
  EXPECT_TRUE(set.IsPointingTo(reg[7], alloca2));
  EXPECT_TRUE(set.IsPointingTo(reg[7], alloca4));

  // %8 should point to external, since it points to the superset of %0 and %1
  EXPECT_TRUE(set.IsPointingToExternal(reg[8]));
  // %8 may also point to alloca4
  EXPECT_LE(set.GetPointsToSet(reg[8]).Size(), 1);
  EXPECT_TRUE(set.IsPointingTo(reg[8], alloca4));

  // %9 may point to v3
  EXPECT_TRUE(set.IsPointingTo(reg[9], alloca3));

  // Due to the store of %9 into [%8], alloca4 may now point back to alloca3
  EXPECT_LE(set.GetPointsToSet(alloca4).Size(), 1);
  EXPECT_TRUE(set.IsPointingTo(alloca4, alloca3));

  // Also due to the same store, alloca3 might have escaped
  EXPECT_TRUE(set.HasEscaped(alloca3));
  // Due to alloca3 pointing to alloca4, it too should have been marked as escaped
  EXPECT_TRUE(set.HasEscaped(alloca4));
  // Check that the other two allocas haven't escaped
  EXPECT_FALSE(set.HasEscaped(alloca1));
  EXPECT_FALSE(set.HasEscaped(alloca2));

  // Make sure only the escaped allocas are marked as pointing to external
  EXPECT_FALSE(set.IsPointingToExternal(alloca1));
  EXPECT_FALSE(set.IsPointingToExternal(alloca2));
  EXPECT_TRUE(set.IsPointingToExternal(alloca3));
  EXPECT_TRUE(set.IsPointingToExternal(alloca4));

  // %10 should also point to external, since it might have been loaded from external
  EXPECT_TRUE(set.IsPointingToExternal(reg[10]));
}

TEST(PointerObjectSetTests, TestPointerObjectConstraintSetSolveNaive)
{
  using Configuration = jlm::llvm::aa::Andersen::Configuration;
  TestPointerObjectConstraintSetSolve<Configuration::Solver::Naive>();
}

TEST(PointerObjectSetTests, TestPointerObjectConstraintSetSolveWorklist)
{
  using Configuration = jlm::llvm::aa::Andersen::Configuration;

  auto allConfigs = jlm::llvm::aa::Andersen::Configuration::GetAllConfigurations();
  for (const auto & config : allConfigs)
  {
    // Ignore all configs that enable features that do not affect SolveUsingWorklist()
    if (config.GetSolver() != jlm::llvm::aa::Andersen::Configuration::Solver::Worklist)
      continue;
    if (config.IsOfflineVariableSubstitutionEnabled())
      continue;
    if (config.IsOfflineConstraintNormalizationEnabled())
      continue;

    TestPointerObjectConstraintSetSolve<Configuration::Solver::Worklist>(
        config.GetWorklistSoliverPolicy(),
        config.IsOnlineCycleDetectionEnabled(),
        config.IsHybridCycleDetectionEnabled(),
        config.IsLazyCycleDetectionEnabled(),
        config.IsDifferencePropagationEnabled(),
        config.IsPreferImplicitPointeesEnabled());
  }
}

TEST(PointerObjectSetTests, TestClonePointerObjectConstraintSet)
{
  using namespace jlm::llvm::aa;
  jlm::llvm::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto register0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput());
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(), true);
  set.AddToPointsToSet(register0, alloca0);

  // Create a dummy register that will point to alloca0 after solving
  const auto dummy0 = set.CreateDummyRegisterPointerObject();

  PointerObjectConstraintSet constraints(set);
  constraints.AddConstraint(SupersetConstraint(dummy0, register0));

  // Make a clone of everything
  auto [setClone, constraintsClone] = constraints.Clone();

  // Modifying the copy doesn't affect the original
  constraintsClone->AddConstraint(LoadConstraint(register0, alloca0));
  EXPECT_EQ(constraintsClone->GetConstraints().size(), 2);
  EXPECT_EQ(constraints.GetConstraints().size(), 1);

  // Solving only affects the PointerObjectSet belonging to that constraint set
  constraints.SolveNaively();
  EXPECT_TRUE(set.GetPointsToSet(dummy0).Contains(alloca0));
  EXPECT_TRUE(setClone->GetPointsToSet(dummy0).IsEmpty());

  constraintsClone->SolveNaively();
  EXPECT_TRUE(setClone->GetPointsToSet(dummy0).Contains(alloca0));
}

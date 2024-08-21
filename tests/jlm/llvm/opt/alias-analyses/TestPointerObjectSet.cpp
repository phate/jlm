/*
 * Copyright 2023, 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>

#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>

#include <cassert>

static bool
StringContains(std::string_view haystack, std::string_view needle)
{
  return haystack.find(needle) != std::string::npos;
}

// Test the flag functions on the PointerObject class
static void
TestFlagFunctions()
{
  using namespace jlm::llvm::aa;

  jlm::tests::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  PointerObjectSet set;
  auto registerPO = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput());

  assert(set.ShouldTrackPointees(registerPO));
  assert(set.IsPointerObjectRegister(registerPO));

  // PointeesEscaping flag
  assert(!set.HasPointeesEscaping(registerPO));
  assert(set.MarkAsPointeesEscaping(registerPO));
  assert(set.HasPointeesEscaping(registerPO));
  // Trying to set the flag again returns false
  assert(!set.MarkAsPointeesEscaping(registerPO));
  assert(set.HasPointeesEscaping(registerPO));

  // PointsToExternal flag. For registers, the two flags are completely independent.
  assert(!set.IsPointingToExternal(registerPO));
  assert(set.MarkAsPointingToExternal(registerPO));
  assert(set.IsPointingToExternal(registerPO));
  // Trying to set the flag again returns false
  assert(!set.MarkAsPointingToExternal(registerPO));
  assert(set.IsPointingToExternal(registerPO));

  // Test that Escaped implies PointsToExternal, for memory objects
  auto allocaPO = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode());

  // alloca may both point
  assert(set.ShouldTrackPointees(allocaPO));
  assert(!set.IsPointerObjectRegister(allocaPO));

  // Escaping means another module can write a pointer to you.
  // This implies another module might override it with pointers to external.
  // It also implies any pointees should also escape
  assert(!set.IsPointingToExternal(allocaPO));
  assert(!set.HasPointeesEscaping(allocaPO));
  assert(set.MarkAsEscaped(allocaPO));
  assert(set.IsPointingToExternal(allocaPO));
  assert(set.HasPointeesEscaping(allocaPO));
  // Already marked with these flags, trying to set them again makes no difference
  assert(!set.MarkAsPointingToExternal(allocaPO));
  assert(!set.MarkAsPointeesEscaping(allocaPO));

  // The analysis should not bother tracking the pointees of lambdas
  auto lambdaPO = set.CreateFunctionMemoryObject(rvsdg.GetLambdaNode());
  assert(!set.ShouldTrackPointees(lambdaPO));
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
  const auto register0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput());
  set.MapRegisterToExistingPointerObject(rvsdg.GetDeltaOutput(), register0);
  const auto dummy0 = set.CreateDummyRegisterPointerObject();

  // For PointerObjects representing MemoryObjects, there is only one Create function
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode());
  const auto malloc0 = set.CreateMallocMemoryObject(rvsdg.GetMallocNode());
  const auto delta0 = set.CreateGlobalMemoryObject(rvsdg.GetDeltaNode());
  const auto lambda0 = set.CreateFunctionMemoryObject(rvsdg.GetLambdaNode());
  const auto import0 = set.CreateImportMemoryObject(rvsdg.GetImportOutput());

  assert(set.NumPointerObjects() == 7);
  assert(set.NumPointerObjectsOfKind(jlm::llvm::aa::PointerObjectKind::Register) == 2);

  assert(set.GetPointerObjectKind(register0) == PointerObjectKind::Register);
  assert(set.GetPointerObjectKind(dummy0) == PointerObjectKind::Register);
  assert(set.GetPointerObjectKind(alloca0) == PointerObjectKind::AllocaMemoryObject);
  assert(set.GetPointerObjectKind(malloc0) == PointerObjectKind::MallocMemoryObject);
  assert(set.GetPointerObjectKind(delta0) == PointerObjectKind::GlobalMemoryObject);
  assert(set.GetPointerObjectKind(lambda0) == PointerObjectKind::FunctionMemoryObject);
  assert(set.GetPointerObjectKind(import0) == PointerObjectKind::ImportMemoryObject);

  // Most pointer objects don't start out as escaped
  assert(!set.HasEscaped(dummy0) && !set.IsPointingToExternal(dummy0));
  assert(!set.HasEscaped(alloca0) && !set.IsPointingToExternal(alloca0));
  assert(!set.HasEscaped(malloc0) && !set.IsPointingToExternal(malloc0));
  assert(!set.HasEscaped(delta0) && !set.IsPointingToExternal(delta0));
  // But import memory objects have always escaped
  assert(set.HasEscaped(import0) && set.IsPointingToExternal(import0));

  // Registers have helper function for looking up existing PointerObjects
  assert(set.GetRegisterPointerObject(rvsdg.GetAllocaOutput()) == register0);
  assert(set.GetRegisterPointerObject(rvsdg.GetDeltaOutput()) == register0);
  assert(set.TryGetRegisterPointerObject(rvsdg.GetDeltaOutput()).value() == register0);

  // Functions have the same, but also in the other direction
  assert(set.GetFunctionMemoryObject(rvsdg.GetLambdaNode()) == lambda0);
  assert(&set.GetLambdaNodeFromFunctionMemoryObject(lambda0) == &rvsdg.GetLambdaNode());

  // The maps can also be accessed directly
  assert(set.GetRegisterMap().at(&rvsdg.GetAllocaOutput()) == register0);
  assert(set.GetRegisterMap().at(&rvsdg.GetDeltaOutput()) == register0);
  assert(set.GetAllocaMap().at(&rvsdg.GetAllocaNode()) == alloca0);
  assert(set.GetMallocMap().at(&rvsdg.GetMallocNode()) == malloc0);
  assert(set.GetGlobalMap().at(&rvsdg.GetDeltaNode()) == delta0);
  assert(set.GetFunctionMap().LookupKey(&rvsdg.GetLambdaNode()) == lambda0);
  assert(set.GetImportMap().at(&rvsdg.GetImportOutput()) == import0);
}

static void
TestPointerObjectUnification()
{
  using namespace jlm::llvm::aa;

  PointerObjectSet set;
  auto dummy0 = set.CreateDummyRegisterPointerObject();
  auto dummy1 = set.CreateDummyRegisterPointerObject();
  assert(set.IsUnificationRoot(dummy0));

  auto root = set.UnifyPointerObjects(dummy0, dummy1);
  assert(set.GetUnificationRoot(dummy0) == root);
  assert(set.GetUnificationRoot(dummy1) == root);

  // Exactly one of the PointerObjects is the root
  assert((root == dummy0) != (root == dummy1));
  assert(set.IsUnificationRoot(root));

  // Trying to unify again gives the same root
  assert(set.UnifyPointerObjects(dummy0, dummy1) == root);

  auto notRoot = dummy0 + dummy1 - root;
  assert(!set.IsUnificationRoot(notRoot));

  auto dummy2 = set.CreateDummyRegisterPointerObject();
  auto dummy3 = set.CreateDummyRegisterPointerObject();
  set.UnifyPointerObjects(dummy0, dummy2);
  auto newRoot = set.UnifyPointerObjects(dummy1, dummy3);
  assert(set.GetUnificationRoot(dummy0) == newRoot);
  assert(set.GetUnificationRoot(dummy1) == newRoot);
  assert(set.GetUnificationRoot(dummy2) == newRoot);
  assert(set.GetUnificationRoot(dummy3) == newRoot);
}

static void
TestPointerObjectUnificationPointees()
{
  using namespace jlm::llvm::aa;

  jlm::tests::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  PointerObjectSet set;
  auto lambda0 = set.CreateFunctionMemoryObject(rvsdg.GetLambdaNode());
  auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode());
  auto delta0 = set.CreateGlobalMemoryObject(rvsdg.GetDeltaNode());

  set.AddToPointsToSet(alloca0, lambda0);
  assert(set.GetPointsToSet(alloca0).Size() == 1);
  assert(set.GetPointsToSet(delta0).Size() == 0);

  set.UnifyPointerObjects(delta0, alloca0);

  // Now both should share points-to sets
  assert(set.GetPointsToSet(alloca0).Contains(lambda0));
  assert(set.GetPointsToSet(delta0).Contains(lambda0));

  // Marking one as pointing to external marks all as pointing to external
  assert(set.MarkAsPointingToExternal(alloca0));
  assert(set.IsPointingToExternal(alloca0));
  assert(set.IsPointingToExternal(delta0));

  // Marking one as pointees escaping marks all as pointees escaping
  assert(set.MarkAsPointeesEscaping(delta0));
  assert(set.HasPointeesEscaping(alloca0));
  assert(set.HasPointeesEscaping(delta0));

  // Adding a new pointee adds it to all members
  auto import0 = set.CreateImportMemoryObject(rvsdg.GetImportOutput());
  assert(set.AddToPointsToSet(delta0, import0));
  assert(set.GetPointsToSet(alloca0).Contains(import0));

  // Escaping is not shared within the unification
  assert(set.MarkAsEscaped(delta0));
  assert(set.HasEscaped(delta0));
  assert(!set.HasEscaped(alloca0));
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
}

static void
TestClonePointerObjectSet()
{
  using namespace jlm::llvm::aa;
  jlm::tests::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto register0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput());
  const auto dummy0 = set.CreateDummyRegisterPointerObject();
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode());
  const auto malloc0 = set.CreateMallocMemoryObject(rvsdg.GetMallocNode());
  const auto delta0 = set.CreateGlobalMemoryObject(rvsdg.GetDeltaNode());
  const auto lambda0 = set.CreateFunctionMemoryObject(rvsdg.GetLambdaNode());
  const auto import0 = set.CreateImportMemoryObject(rvsdg.GetImportOutput());

  set.AddToPointsToSet(register0, alloca0);
  set.UnifyPointerObjects(delta0, import0);

  auto clonedSet = set.Clone();

  // All mappings are identical, since PointerObjects are referenced by index
  assert(clonedSet->NumPointerObjects() == set.NumPointerObjects());
  assert(clonedSet->GetRegisterMap() == set.GetRegisterMap());
  assert(clonedSet->GetAllocaMap() == set.GetAllocaMap());
  assert(clonedSet->GetMallocMap() == set.GetMallocMap());
  assert(clonedSet->GetGlobalMap() == set.GetGlobalMap());
  assert(clonedSet->GetFunctionMap() == set.GetFunctionMap());
  assert(clonedSet->GetImportMap() == set.GetImportMap());

  // In the cloned set, each points-to set should be identical
  assert(clonedSet->GetPointsToSet(register0) == set.GetPointsToSet(register0));

  // Unifications are maintained
  assert(clonedSet->GetUnificationRoot(delta0) == clonedSet->GetUnificationRoot(import0));

  // Additional changes only affect the set they are applied to
  set.AddToPointsToSet(register0, lambda0);
  assert(!clonedSet->GetPointsToSet(register0).Contains(lambda0));

  clonedSet->UnifyPointerObjects(delta0, dummy0);
  assert(set.GetUnificationRoot(delta0) != set.GetUnificationRoot(dummy0));

  set.MarkAsPointingToExternal(malloc0);
  assert(!clonedSet->IsPointingToExternal(malloc0));
}

// Test the SupersetConstraint's Apply function
static void
TestSupersetConstraint()
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
  const auto reg2 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(2));

  set.AddToPointsToSet(reg0, alloca0);
  set.AddToPointsToSet(reg1, alloca1);
  set.AddToPointsToSet(reg2, alloca2);

  // Make alloca0 point to everything reg1 points to
  SupersetConstraint c1(alloca0, reg1);
  assert(c1.ApplyDirectly(set));
  while (c1.ApplyDirectly(set))
    ;
  // For now this only makes alloca0 point to alloca1
  assert(set.GetPointsToSet(alloca0).Size() == 1);
  assert(set.GetPointsToSet(alloca0).Contains(alloca1));

  // Make reg1 point to everything reg2 points to
  SupersetConstraint c2(reg1, reg2);
  assert(c2.ApplyDirectly(set));
  while (c2.ApplyDirectly(set))
    ;
  // This makes alloca2 a member of P(reg1)
  assert(set.GetPointsToSet(reg1).Contains(alloca2));

  // Apply c1 again
  assert(c1.ApplyDirectly(set));
  while (c1.ApplyDirectly(set))
    ;
  // Now alloca0 should also point to alloca2
  assert(set.GetPointsToSet(alloca0).Size() == 2);
  assert(set.GetPointsToSet(alloca0).Contains(alloca2));

  // Make reg2 point to external, and propagate through constraints
  set.MarkAsPointingToExternal(reg2);
  assert(c2.ApplyDirectly(set));
  while (c2.ApplyDirectly(set))
    ;
  assert(c1.ApplyDirectly(set));
  while (c1.ApplyDirectly(set))
    ;
  // Now both reg1 and alloca0 may point to external
  assert(set.IsPointingToExternal(reg1));
  assert(set.IsPointingToExternal(alloca0));
}

static void
TestStoreConstraintDirectly()
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
  const auto reg2 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(2));

  set.AddToPointsToSet(reg0, alloca0);
  set.AddToPointsToSet(reg1, alloca1);
  set.AddToPointsToSet(reg2, alloca2);

  // Add *alloca0 = reg2, which will do nothing, since alloca0 can't be pointing to anything yet
  StoreConstraint c1(alloca0, reg2);
  assert(!c1.ApplyDirectly(set));

  // This means *reg0 = reg1, and as we know, reg0 points to alloca0
  // This should make alloca0 point to anything reg1 points to, aka alloca1
  StoreConstraint c2(reg0, reg1);
  assert(c2.ApplyDirectly(set));
  while (c2.ApplyDirectly(set))
    ;
  assert(set.GetPointsToSet(alloca0).Size() == 1);
  assert(set.GetPointsToSet(alloca0).Contains(alloca1));

  // Do c1 again, now that alloca0 points to alloca1
  assert(c1.ApplyDirectly(set));
  while (c1.ApplyDirectly(set))
    ;
  assert(set.GetPointsToSet(alloca1).Size() == 1);
  assert(set.GetPointsToSet(alloca1).Contains(alloca2));
}

static void
TestLoadConstraintDirectly()
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
  const auto reg2 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(2));

  set.AddToPointsToSet(reg0, alloca0);
  set.AddToPointsToSet(reg1, alloca1);
  set.AddToPointsToSet(reg2, alloca2);

  // Makes reg1 = *reg0, where reg0 currently just points to alloca0
  // Since alloca0 currently has no pointees, this does nothing
  LoadConstraint c1(reg1, reg0);
  assert(!c1.ApplyDirectly(set));

  // Make alloca0 point to something: alloca2
  set.AddToPointsToSet(alloca0, alloca2);
  // The constraint now makes a difference
  assert(c1.ApplyDirectly(set));
  while (c1.ApplyDirectly(set))
    ;
  assert(set.GetPointsToSet(reg1).Size() == 2);
  assert(set.GetPointsToSet(reg1).Contains(alloca2));
}

static void
TestEscapedFunctionConstraint()
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

  bool result = EscapedFunctionConstraint::PropagateEscapedFunctionsDirectly(set);

  // Nothing has happened yet, since the exported function is yet to be marked as escaped
  assert(!result);
  assert(!set.HasEscaped(localFunctionRegisterPO));
  assert(!set.HasEscaped(exportedFunctionPO));

  set.MarkAsEscaped(exportedFunctionPO);

  // Use both EscapedFunctionConstraint and EscapeFlagConstraint to propagate flags
  result = EscapedFunctionConstraint::PropagateEscapedFunctionsDirectly(set);
  result &= EscapeFlagConstraint::PropagateEscapedFlagsDirectly(set);

  // Now the local function has been marked as escaped as well, since it is the return value
  assert(result);
  assert(set.HasEscaped(localFunctionPO));
}

static void
TestFunctionCallConstraint()
{
  using namespace jlm::llvm::aa;

  jlm::tests::CallTest1 rvsdg;
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto lambdaF = set.CreateFunctionMemoryObject(*rvsdg.lambda_f);
  const auto lambdaFRegister = set.CreateRegisterPointerObject(*rvsdg.lambda_f->output());
  const auto lambdaFArgumentX = set.CreateRegisterPointerObject(*rvsdg.lambda_f->fctargument(0));
  const auto lambdaFArgumentY = set.CreateRegisterPointerObject(*rvsdg.lambda_f->fctargument(1));
  const auto allocaX = set.CreateAllocaMemoryObject(*rvsdg.alloca_x);
  const auto allocaY = set.CreateAllocaMemoryObject(*rvsdg.alloca_y);
  const auto allocaXRegister = set.CreateRegisterPointerObject(*rvsdg.alloca_x->output(0));
  const auto allocaYRegister = set.CreateRegisterPointerObject(*rvsdg.alloca_y->output(0));

  // Associate the allocas and lambda with their output
  set.AddToPointsToSet(lambdaFRegister, lambdaF);
  set.AddToPointsToSet(allocaXRegister, allocaX);
  set.AddToPointsToSet(allocaYRegister, allocaY);

  FunctionCallConstraint c(lambdaFRegister, rvsdg.CallF());
  assert(c.ApplyDirectly(set));
  while (c.ApplyDirectly(set))
    ;

  // The arguments to f should now be pointing to exactly the corresponding allocas
  assert(set.GetPointsToSet(lambdaFArgumentX).Contains(allocaX));
  assert(set.GetPointsToSet(lambdaFArgumentY).Contains(allocaY));
  assert(!set.GetPointsToSet(lambdaFArgumentX).Contains(allocaY));
  assert(!set.GetPointsToSet(lambdaFArgumentY).Contains(allocaX));
}

static void
TestAddPointsToExternalConstraint()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(2);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  const auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  const auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1));
  const auto reg1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));

  PointerObjectConstraintSet constraints(set);
  constraints.AddPointerPointeeConstraint(reg0, alloca0);
  constraints.AddPointerPointeeConstraint(reg1, alloca1);

  constraints.AddPointsToExternalConstraint(reg0);
  constraints.SolveNaively();

  // Make sure only reg0 points to external, and nothing has escaped
  assert(set.IsPointingToExternal(reg0));
  assert(!set.IsPointingToExternal(reg1));
  assert(!set.IsPointingToExternal(alloca0));
  assert(!set.IsPointingToExternal(alloca1));

  assert(!set.HasEscaped(reg0));
  assert(!set.HasEscaped(reg1));
  assert(!set.HasEscaped(alloca0));
  assert(!set.HasEscaped(alloca1));

  // Add a *reg0 = reg1 store
  constraints.AddConstraint(StoreConstraint(reg0, reg1));
  constraints.SolveNaively();

  // Now alloca1 is marked as escaped, due to being written to a pointer that might point to
  // external
  assert(set.HasEscaped(alloca1));
  // The other alloca has not escaped
  assert(!set.HasEscaped(alloca0));
}

static void
TestAddRegisterContentEscapedConstraint()
{
  using namespace jlm::llvm::aa;

  jlm::tests::NAllocaNodesTest rvsdg(2);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  const auto reg0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  const auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1));
  const auto reg1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));

  PointerObjectConstraintSet constraints(set);
  constraints.AddPointerPointeeConstraint(reg0, alloca0);
  constraints.AddPointerPointeeConstraint(reg1, alloca1);

  // return reg0, making alloca0 escapce
  constraints.AddRegisterContentEscapedConstraint(reg0);
  constraints.SolveNaively();

  // Make sure only alloca0 has escaped
  assert(set.HasEscaped(alloca0));
  assert(!set.HasEscaped(alloca1));

  // Add a alloca0 = reg1 store
  constraints.AddConstraint(StoreConstraint(reg0, reg1));
  constraints.SolveNaively();

  // Now both are marked as escaped
  assert(set.HasEscaped(alloca0));
  assert(set.HasEscaped(alloca1));
}

static void
TestDrawSubsetGraph()
{
  using namespace jlm::llvm::aa;
  using namespace jlm::util;
  jlm::tests::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  // Arrange
  PointerObjectSet set;
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode());
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

  const auto import0 = set.CreateImportMemoryObject(rvsdg.GetImportOutput());

  PointerObjectConstraintSet constraints(set);
  constraints.AddPointsToExternalConstraint(nonRoot);
  constraints.AddPointerPointeeConstraint(allocaReg0, alloca0);
  constraints.AddConstraint(SupersetConstraint(import0, allocaReg0));
  constraints.AddConstraint(StoreConstraint(storePointer, storeValue));
  constraints.AddConstraint(LoadConstraint(loadValue, loadPointer));

  // Act
  GraphWriter writer;
  auto & graph = constraints.DrawSubsetGraph(writer);

  // Assert
  assert(graph.NumNodes() == set.NumPointerObjects());

  // Check that the unified node that is not the root, contains the index of the root
  assert(StringContains(graph.GetNode(nonRoot).GetLabel(), "#" + std::to_string(root)));

  // Check that the unification root's label indicates pointing to external
  assert(StringContains(graph.GetNode(root).GetLabel(), "{+}"));

  // Check that allocaReg0 points to alloca0
  assert(StringContains(graph.GetNode(allocaReg0).GetLabel(), strfmt("{", alloca0, "}")));

  // Check that a regular edge connects allocaReg0 to the importNode
  auto * supersetEdge = graph.GetEdgeBetween(graph.GetNode(allocaReg0), graph.GetNode(import0));
  assert(supersetEdge);
  assert(supersetEdge->IsDirected());
  assert(supersetEdge->GetAttributeOr("style", "solid") == "solid");

  // Check that a store edge connects storeValue to storePointer
  auto * storeEdge = graph.GetEdgeBetween(graph.GetNode(storeValue), graph.GetNode(storePointer));
  assert(storeEdge);
  assert(storeEdge->IsDirected());
  assert(storeEdge->GetAttributeOr("style", Edge::Style::Dashed) == Edge::Style::Dashed);
  assert(StringContains(storeEdge->GetAttribute("arrowhead"), "dot"));

  // Check that a load edge connects loadPointer to loadValue
  auto * loadEdge = graph.GetEdgeBetween(graph.GetNode(loadPointer), graph.GetNode(loadValue));
  assert(loadEdge);
  assert(loadEdge->IsDirected());
  assert(loadEdge->GetAttributeOr("style", Edge::Style::Dashed) == Edge::Style::Dashed);
  assert(StringContains(loadEdge->GetAttribute("arrowtail"), "dot"));

  // Check that the function contains the word "function0"
  auto & functionNode = graph.GetNode(function0);
  assert(StringContains(functionNode.GetLabel(), "function0"));
  // Since functions don't track pointees, they should have NOTRACK
  assert(StringContains(functionNode.GetLabel(), "NOTRACK"));
  // They should also both point to external, and escape all pointees
  assert(StringContains(functionNode.GetLabel(), "{+}e"));

  // Check that the import PointerObject has a fill color, since it has escaped
  auto & importNode = graph.GetNode(import0);
  assert(importNode.HasAttribute("fillcolor"));
}

// Tests crating a ConstraintSet with multiple different constraints and calling Solve()
template<bool useWorklist, typename... Args>
static void
TestPointerObjectConstraintSetSolve(Args... args)
{
  using namespace jlm::llvm::aa;

  // Create a graph with 11 different registers, and 4 allocas.
  jlm::tests::NAllocaNodesTest rvsdg(4);
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
  const auto alloca1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  const auto alloca2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1));
  const auto alloca3 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2));
  const auto alloca4 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(3));

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
  if constexpr (useWorklist)
  {
    constraints.SolveUsingWorklist(args...);
  }
  else
  {
    static_assert(sizeof...(args) == 0, "The naive solver takes no arguments");
    constraints.SolveNaively();
  }

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
  assert(set.IsPointingToExternal(reg[8]));
  // %8 may also point to alloca4
  assert(set.GetPointsToSet(reg[8]).Size() == 1);
  assert(set.GetPointsToSet(reg[8]).Contains(alloca4));

  // %9 may point to v3
  assert(set.GetPointsToSet(reg[9]).Contains(alloca3));

  // Due to the store of %9 into [%8], alloca4 may now point back to alloca3
  assert(set.GetPointsToSet(alloca4).Size() == 1);
  assert(set.GetPointsToSet(alloca4).Contains(alloca3));

  // Also due to the same store, alloca3 might have escaped
  assert(set.HasEscaped(alloca3));
  // Due to alloca3 pointing to alloca4, it too should have been marked as escaped
  assert(set.HasEscaped(alloca4));
  // Check that the other two allocas haven't escaped
  assert(!set.HasEscaped(alloca1));
  assert(!set.HasEscaped(alloca2));

  // Make sure only the escaped allocas are marked as pointing to external
  assert(!set.IsPointingToExternal(alloca1));
  assert(!set.IsPointingToExternal(alloca2));
  assert(set.IsPointingToExternal(alloca3));
  assert(set.IsPointingToExternal(alloca4));

  // %10 should also point to external, since it might have been loaded from external
  assert(set.IsPointingToExternal(reg[10]));
}

static void
TestClonePointerObjectConstraintSet()
{
  using namespace jlm::llvm::aa;
  jlm::tests::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  PointerObjectSet set;
  const auto register0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput());
  const auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode());
  set.AddToPointsToSet(register0, alloca0);

  // Create a dummy register that will point to alloca0 after solving
  const auto dummy0 = set.CreateDummyRegisterPointerObject();

  PointerObjectConstraintSet constraints(set);
  constraints.AddConstraint(SupersetConstraint(dummy0, register0));

  // Make a clone of everything
  auto [setClone, constraintsClone] = constraints.Clone();

  // Modifying the copy doesn't affect the original
  constraintsClone->AddConstraint(LoadConstraint(register0, alloca0));
  assert(constraintsClone->GetConstraints().size() == 2);
  assert(constraints.GetConstraints().size() == 1);

  // Solving only affects the PointerObjectSet belonging to that constraint set
  constraints.SolveNaively();
  assert(set.GetPointsToSet(dummy0).Contains(alloca0));
  assert(setClone->GetPointsToSet(dummy0).IsEmpty());

  constraintsClone->SolveNaively();
  assert(setClone->GetPointsToSet(dummy0).Contains(alloca0));
}

static int
TestPointerObjectSet()
{
  TestFlagFunctions();
  TestCreatePointerObjects();
  TestPointerObjectUnification();
  TestPointerObjectUnificationPointees();
  TestAddToPointsToSet();
  TestMakePointsToSetSuperset();
  TestClonePointerObjectSet();
  TestSupersetConstraint();
  TestStoreConstraintDirectly();
  TestLoadConstraintDirectly();
  TestEscapedFunctionConstraint();
  TestFunctionCallConstraint();
  TestAddPointsToExternalConstraint();
  TestAddRegisterContentEscapedConstraint();
  TestDrawSubsetGraph();
  TestPointerObjectConstraintSetSolve<false>();

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

    TestPointerObjectConstraintSetSolve<true>(
        config.GetWorklistSoliverPolicy(),
        config.IsOnlineCycleDetectionEnabled(),
        config.IsHybridCycleDetectionEnabled(),
        config.IsLazyCycleDetectionEnabled(),
        config.IsDifferencePropagationEnabled());
  }

  TestClonePointerObjectConstraintSet();
  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestPointerObjectSet", TestPointerObjectSet)

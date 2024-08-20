/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <TestRvsdgs.hpp>

#include <test-registry.hpp>

#include <jlm/llvm/opt/alias-analyses/DifferencePropagation.hpp>
#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>

#include <cassert>

static int
TestTracksDifferences()
{
  using namespace jlm;
  using namespace jlm::llvm::aa;

  // Arrange
  tests::NAllocaNodesTest rvsdg(4);
  rvsdg.InitializeTest();

  PointerObjectSet set;
  auto r0 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(0));
  auto r1 = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput(1));
  auto a0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(0));
  auto a1 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(1));
  auto a2 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(2));
  auto a3 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(3));

  // Let r0 -> a0 and r0 -> a3 before difference tracking even begins
  set.AddToPointsToSet(r0, a0);
  set.AddToPointsToSet(r0, a3);

  // Act
  DifferencePropagation differencePropagation(set);
  differencePropagation.Initialize();

  // Assert
  assert(differencePropagation.GetNewPointees(r0) == (util::HashSet<PointerObjectIndex>{ a0, a3 }));

  // Act 2 - add another pointer/pointee relation: r1 -> a1
  differencePropagation.AddToPointsToSet(r1, a1);

  // Assert that a1 is a new pointee of r1
  assert(differencePropagation.GetNewPointees(r1) == (util::HashSet<PointerObjectIndex>{ a1 }));

  // Act 3 - clear difference tracking for r1
  differencePropagation.ClearNewPointees(r1);

  // Assert r1 no longer has any new pointees
  assert(differencePropagation.GetNewPointees(r1).IsEmpty());
  // r0 still has new pointees
  assert(!differencePropagation.GetNewPointees(r0).IsEmpty());

  // Act 4 - add more pointees to r1,
  bool new0 = differencePropagation.AddToPointsToSet(r1, a0);
  bool new1 = differencePropagation.AddToPointsToSet(r1, a1); // not a new pointee
  bool new2 = differencePropagation.AddToPointsToSet(r1, a2);

  // Assert that only a0 and a2 were new
  assert(new0 && !new1 && new2);
  assert(differencePropagation.GetNewPointees(r1) == (util::HashSet<PointerObjectIndex>{ a0, a2 }));

  // Act 5 - make r0 point to a superset of r1, making r0 now point to a0, a1, a2, a3
  // First mark the existing pointees of r0 (a0 and a3) as seen
  differencePropagation.ClearNewPointees(r0);
  differencePropagation.MakePointsToSetSuperset(r0, r1);

  // Assert that only a1 and a2 are new to r0, as it has already marked a0 and a3 as seen
  assert(differencePropagation.GetNewPointees(r0) == (util::HashSet<PointerObjectIndex>{ a1, a2 }));

  // Act 6 - give nodes r0 and r1 flags
  set.MarkAsPointeesEscaping(r0);
  set.MarkAsPointingToExternal(r1);

  // Assert that the flags are new, but only if they actually have the flag
  assert(differencePropagation.PointeesEscapeIsNew(r0));
  assert(!differencePropagation.PointsToExternalIsNew(r0));
  assert(!differencePropagation.PointeesEscapeIsNew(r1));
  assert(differencePropagation.PointsToExternalIsNew(r1));

  // Act 7 - mark flags as seen
  differencePropagation.MarkPointeesEscapeAsHandled(r0);
  differencePropagation.MarkPointsToExternalAsHandled(r1);

  // Assert that the flags are no longer new
  assert(!differencePropagation.PointeesEscapeIsNew(r0));
  assert(!differencePropagation.PointsToExternalIsNew(r1));

  // Act 6 - unify 0 and 1
  // After unification, any pointee or flag that is new to either node becomes new to the union
  auto root = set.UnifyPointerObjects(r0, r1);
  auto nonRoot = r0 + r1 - root;
  differencePropagation.OnPointerObjectsUnified(root, nonRoot);

  // Assert that all pointees that were new to either node, are also new to the root
  // a0 and a2 were still marked as new to node r1 at the time of unification.
  // a3 is not new to r0, but r1 has never seen it, so it must be regarded as new by the union.
  util::HashSet<PointerObjectIndex> subset{ a0, a2, a3 };
  assert(subset.IsSubsetOf(differencePropagation.GetNewPointees(root)));

  // Neither flag has been seen by both nodes, so they are both new to the unification
  assert(differencePropagation.PointeesEscapeIsNew(root));
  assert(differencePropagation.PointsToExternalIsNew(root));

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestDifferencePropagation-TestTracksDifferences",
    TestTracksDifferences)

/*
 * Copyright 2023, 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <TestRvsdgs.hpp>

#include <test-registry.hpp>

#include <jlm/llvm/opt/alias-analyses/LazyCycleDetection.hpp>
#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>
#include <jlm/util/HashSet.hpp>

#include <cassert>
#include <vector>

static void
TestUnifiesCycles()
{
  using namespace jlm;
  using namespace jlm::llvm::aa;

  // Arrange
  PointerObjectSet set;
  for (int i = 0; i < 6; i++)
  {
    (void)set.CreateDummyRegisterPointerObject();
  }

  // Create a graph that looks like
  //   --> 1 --> 2 --> 3
  //  /          |
  // 0           |
  //  \          V
  //   --> 5 --> 4
  std::vector<util::HashSet<PointerObjectIndex>> successors(set.NumPointerObjects());
  successors[0].Insert(1);
  successors[1].Insert(2);
  successors[2].Insert(3);
  successors[2].Insert(4);
  successors[0].Insert(5);
  successors[5].Insert(4);

  auto GetSuccessors = [&](PointerObjectIndex i)
  {
    assert(set.IsUnificationRoot(i));
    return successors[i].Items();
  };

  auto UnifyPointerObjects = [&](PointerObjectIndex a, PointerObjectIndex b)
  {
    assert(set.IsUnificationRoot(a));
    assert(set.IsUnificationRoot(b));
    assert(a != b);
    auto newRoot = set.UnifyPointerObjects(a, b);
    auto notRoot = a + b - newRoot;

    successors[newRoot].UnionWith(successors[notRoot]);
    return newRoot;
  };

  LazyCycleDetector lcd(set, GetSuccessors, UnifyPointerObjects);
  lcd.Initialize();

  // Act 1 - an edge that is not a part of a cycle
  lcd.OnPropagatedNothing(0, 1);

  // Assert that nothing happened
  assert(lcd.NumCycleDetectionAttempts() == 1);
  assert(lcd.NumCyclesDetected() == 0);
  assert(lcd.NumCycleUnifications() == 0);

  // Act 2 - Try the same edge again
  lcd.OnPropagatedNothing(0, 1);

  // Assert that the second attempt is ignored
  assert(lcd.NumCycleDetectionAttempts() == 1);
  assert(lcd.NumCyclesDetected() == 0);
  assert(lcd.NumCycleUnifications() == 0);

  // Act 3 - add the edge 3->1 that creates a cycle 3-1-2-3
  successors[3].Insert(1);
  lcd.OnPropagatedNothing(3, 1);

  // Assert that the cycle was found and unified
  assert(lcd.NumCycleDetectionAttempts() == 2);
  assert(lcd.NumCyclesDetected() == 1);
  assert(lcd.NumCycleUnifications() == 2);
  assert(set.GetUnificationRoot(1) == set.GetUnificationRoot(2));
  assert(set.GetUnificationRoot(1) == set.GetUnificationRoot(3));

  // Act 4 - add the edge 4 -> 0, creating two cycles 4-0-5-4 and 4-0-(1/2/3)-4
  successors[4].Insert(0);
  lcd.OnPropagatedNothing(4, 0);

  // Assert that both cycles were found.
  // They are only counted as one cycle, but everything should be unified now
  assert(lcd.NumCyclesDetected() == 2);
  assert(lcd.NumCycleUnifications() == set.NumPointerObjects() - 1);
  for (PointerObjectIndex i = 1; i < set.NumPointerObjects(); i++)
  {
    assert(set.GetUnificationRoot(0) == set.GetUnificationRoot(i));
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestLazyCycleDetection-TestUnifiesCycles",
    TestUnifiesCycles)

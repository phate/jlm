/*
 * Copyright 2023, 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/opt/alias-analyses/LazyCycleDetection.hpp>
#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>
#include <jlm/util/HashSet.hpp>

#include <cassert>
#include <vector>

TEST(LazyCycleDetectionTests, TestUnifiesCycles)
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
  successors[0].insert(1);
  successors[1].insert(2);
  successors[2].insert(3);
  successors[2].insert(4);
  successors[0].insert(5);
  successors[5].insert(4);

  auto GetSuccessors = [&](PointerObjectIndex i)
  {
    EXPECT_TRUE(set.IsUnificationRoot(i));
    return successors[i].Items();
  };

  auto UnifyPointerObjects = [&](PointerObjectIndex a, PointerObjectIndex b)
  {
    EXPECT_TRUE(set.IsUnificationRoot(a));
    EXPECT_TRUE(set.IsUnificationRoot(b));
    EXPECT_NE(a, b);
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
  EXPECT_EQ(lcd.NumCycleDetectionAttempts(), 1);
  EXPECT_EQ(lcd.NumCyclesDetected(), 0);
  EXPECT_EQ(lcd.NumCycleUnifications(), 0);

  // Act 2 - Try the same edge again
  lcd.OnPropagatedNothing(0, 1);

  // Assert that the second attempt is ignored
  EXPECT_EQ(lcd.NumCycleDetectionAttempts(), 1);
  EXPECT_EQ(lcd.NumCyclesDetected(), 0);
  EXPECT_EQ(lcd.NumCycleUnifications(), 0);

  // Act 3 - add the edge 3->1 that creates a cycle 3-1-2-3
  successors[3].insert(1);
  lcd.OnPropagatedNothing(3, 1);

  // Assert that the cycle was found and unified
  EXPECT_EQ(lcd.NumCycleDetectionAttempts(), 2);
  EXPECT_EQ(lcd.NumCyclesDetected(), 1);
  EXPECT_EQ(lcd.NumCycleUnifications(), 2);
  EXPECT_EQ(set.GetUnificationRoot(1), set.GetUnificationRoot(2));
  EXPECT_EQ(set.GetUnificationRoot(1), set.GetUnificationRoot(3));

  // Act 4 - add the edge 4 -> 0, creating two cycles 4-0-5-4 and 4-0-(1/2/3)-4
  successors[4].insert(0);
  lcd.OnPropagatedNothing(4, 0);

  // Assert that both cycles were found.
  // They are only counted as one cycle, but everything should be unified now
  EXPECT_EQ(lcd.NumCyclesDetected(), 2);
  EXPECT_EQ(lcd.NumCycleUnifications(), set.NumPointerObjects() - 1);
  for (PointerObjectIndex i = 1; i < set.NumPointerObjects(); i++)
  {
    EXPECT_EQ(set.GetUnificationRoot(0), set.GetUnificationRoot(i));
  }
}

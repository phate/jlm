/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/util/HashSet.hpp>

#include <memory>

TEST(HashSetTests, TestInt)
{
  jlm::util::HashSet<int> hashSet({ 0, 1, 2, 3, 4, 5, 6, 7 });

  EXPECT_TRUE(hashSet.Size());
  EXPECT_TRUE(hashSet.Contains(3));
  EXPECT_FALSE(hashSet.Contains(8));

  EXPECT_TRUE(hashSet.insert(8));
  EXPECT_TRUE(hashSet.Contains(8));
  EXPECT_FALSE(hashSet.insert(8));

  EXPECT_TRUE(hashSet.Remove(1));
  EXPECT_FALSE(hashSet.Contains(1));
  EXPECT_FALSE(hashSet.Remove(1));

  int sum = 0;
  for (auto & item : hashSet.Items())
    sum += item;
  EXPECT_EQ(sum, 0 + 2 + 3 + 4 + 5 + 6 + 7 + 8);

  jlm::util::HashSet<int> hashSet2({ 8, 9 });
  hashSet.UnionWith(hashSet2);
  EXPECT_EQ(hashSet.Size(), 9u);

  auto numRemoved = hashSet.RemoveWhere(
      [](int n) -> bool
      {
        return n % 2 != 0;
      });
  EXPECT_EQ(numRemoved, 4u);

  hashSet.Clear();
  EXPECT_EQ(hashSet.Size(), 0u);
}

TEST(HashSetTests, TestUniquePointer)
{
  jlm::util::HashSet<std::unique_ptr<int>> hashSet;

  hashSet.insert(std::make_unique<int>(0));
  EXPECT_EQ(hashSet.Size(), 1u);

  hashSet.Remove(std::make_unique<int>(0));
  EXPECT_EQ(hashSet.Size(), 1u);

  hashSet.Clear();
  EXPECT_EQ(hashSet.Size(), 0u);
}

TEST(HashSetTests, TestPair)
{
  jlm::util::HashSet<std::pair<int, int>> hashSet{ { 1, 10 }, { 5, 50 } };

  // Inserting new value
  auto result = hashSet.insert({ 7, 70 });
  EXPECT_TRUE(result);
  EXPECT_EQ(hashSet.Size(), 3u);

  // Try inserting already inserted
  result = hashSet.insert({ 1, 10 });
  EXPECT_FALSE(result);
  EXPECT_EQ(hashSet.Size(), 3u);

  // Contains is only true in the correct order
  EXPECT_TRUE(hashSet.Contains({ 5, 50 }));
  EXPECT_FALSE(hashSet.Contains({ 50, 5 }));

  // Removing works
  result = hashSet.Remove({ 5, 50 });
  EXPECT_TRUE(result);
  EXPECT_EQ(hashSet.Size(), 2u);
  result = hashSet.Remove({ 5, 50 });
  EXPECT_FALSE(result);
  EXPECT_EQ(hashSet.Size(), 2u);
}

TEST(HashSetTests, TestIsSubsetOf)
{
  jlm::util::HashSet<int> set12({ 1, 2 });
  jlm::util::HashSet<int> set123({ 1, 2, 3 });
  jlm::util::HashSet<int> set1234({ 1, 2, 3, 4 });

  EXPECT_TRUE(set12.IsSubsetOf(set12));
  EXPECT_TRUE(set12.IsSubsetOf(set123));
  EXPECT_TRUE(set12.IsSubsetOf(set1234));
  EXPECT_FALSE(set123.IsSubsetOf(set12));
  EXPECT_TRUE(set123.IsSubsetOf(set1234));
  EXPECT_FALSE(set1234.IsSubsetOf(set12));
  EXPECT_FALSE(set1234.IsSubsetOf(set123));
}

TEST(HashSetTests, TestUnionWith)
{
  using namespace jlm::util;

  HashSet<int> set12({ 1, 2 });
  HashSet<int> set123({ 1, 2, 3 });
  HashSet<int> set45({ 4, 5 });

  // Unioning with a subset should not change anything
  bool result = set123.UnionWith(set12);
  EXPECT_FALSE(result);
  EXPECT_EQ(set123.Size(), 3u);

  // Putting {1, 2, 3} into {1, 2} should make it grow
  result = set12.UnionWith(set123);
  EXPECT_TRUE(result);
  EXPECT_EQ(set12.Size(), 3u);
  EXPECT_EQ(set12, set123);

  // Unioning again does nothing
  result = set12.UnionWith(set123);
  EXPECT_FALSE(result);

  // Test union and clear
  result = set45.UnionWithAndClear(set123);
  EXPECT_TRUE(result);
  EXPECT_EQ(set45.Size(), 5u);
  EXPECT_TRUE(set123.IsEmpty());
}

TEST(HashSetTests, TestIntersectWith)
{
  using namespace jlm::util;

  HashSet<int> set12({ 1, 2 });
  HashSet<int> set123({ 1, 2, 3 });
  HashSet<int> set45({ 4, 5 });

  set123.IntersectWith(set12);
  EXPECT_EQ(set123, set12);

  set123.IntersectWith(set45);
  EXPECT_EQ(set123.Size(), 0u);

  set123.insert(1);
  set123.insert(2);
  set123.insert(3);
  set123.IntersectWithAndClear(set12);

  EXPECT_EQ(set123.Size(), 2u);
  EXPECT_EQ(set12.Size(), 0u);
}

TEST(HashSetTests, TestDifferenceWith)
{
  using namespace jlm::util;

  HashSet<int> set12({ 1, 2 });
  HashSet<int> set123({ 1, 2, 3 });
  HashSet<int> set45({ 4, 5 });

  const auto set12Copy = set12;

  set123.DifferenceWith(set12); // {1, 2, 3} - {1, 2}
  EXPECT_EQ(set123.Size(), 1u);
  EXPECT_TRUE(set123.Contains(3));

  // set12 was not touched
  EXPECT_EQ(set12, set12Copy);

  // Create the set {0, 1, 3}
  set123.insert(0);
  set123.insert(1);

  set12.DifferenceWith(set123); // {1, 2} - {0, 1, 3}
  EXPECT_EQ(set12.Size(), 1u);
  EXPECT_TRUE(set12.Contains(2));

  // Difference with other sets becomes empty
  set45.DifferenceWith(set123);
  set45.DifferenceWith(set12);
  EXPECT_EQ(set45.Size(), 2u);

  // We handle the case where both sets are the same set without crashing
  set45.DifferenceWith(set45);
  EXPECT_TRUE(set45.IsEmpty());
}

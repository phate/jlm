/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/HashSet.hpp>

#include <cassert>
#include <memory>

static int
TestInt()
{
  jlm::util::HashSet<int> hashSet({ 0, 1, 2, 3, 4, 5, 6, 7 });

  assert(hashSet.Size());
  assert(hashSet.Contains(3));
  assert(!hashSet.Contains(8));

  assert(hashSet.Insert(8));
  assert(hashSet.Contains(8));
  assert(!hashSet.Insert(8));

  assert(hashSet.Remove(1));
  assert(!hashSet.Contains(1));
  assert(!hashSet.Remove(1));

  int sum = 0;
  for (auto & item : hashSet.Items())
    sum += item;
  assert(sum == (0 + 2 + 3 + 4 + 5 + 6 + 7 + 8));

  jlm::util::HashSet<int> hashSet2({ 8, 9 });
  hashSet.UnionWith(hashSet2);
  assert(hashSet.Size() == 9);

  auto numRemoved = hashSet.RemoveWhere(
      [](int n) -> bool
      {
        return n % 2 != 0;
      });
  assert(numRemoved == 4);

  hashSet.Clear();
  assert(hashSet.Size() == 0);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestHashSet-TestInt", TestInt)

static int
TestUniquePointer()
{
  jlm::util::HashSet<std::unique_ptr<int>> hashSet;

  hashSet.Insert(std::make_unique<int>(0));
  assert(hashSet.Size() == 1);

  hashSet.Remove(std::make_unique<int>(0));
  assert(hashSet.Size() == 1);

  hashSet.Clear();
  assert(hashSet.Size() == 0);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestHashSet-TestUniquePointer", TestUniquePointer)

static int
TestPair()
{
  jlm::util::HashSet<std::pair<int, int>> hashSet{ { 1, 10 }, { 5, 50 } };

  // Inserting new value
  auto result = hashSet.Insert({ 7, 70 });
  assert(result && hashSet.Size() == 3);

  // Try inserting already inserted
  result = hashSet.Insert({ 1, 10 });
  assert(!result && hashSet.Size() == 3);

  // Contains is only true in the correct order
  assert(hashSet.Contains({ 5, 50 }));
  assert(!hashSet.Contains({ 50, 5 }));

  // Removing works
  result = hashSet.Remove({ 5, 50 });
  assert(result && hashSet.Size() == 2);
  result = hashSet.Remove({ 5, 50 });
  assert(!result && hashSet.Size() == 2);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestHashSet-TestPair", TestPair)

static int
TestIsSubsetOf()
{
  jlm::util::HashSet<int> set12({ 1, 2 });
  jlm::util::HashSet<int> set123({ 1, 2, 3 });
  jlm::util::HashSet<int> set1234({ 1, 2, 3, 4 });

  assert(set12.IsSubsetOf(set12));
  assert(set12.IsSubsetOf(set123));
  assert(set12.IsSubsetOf(set1234));
  assert(!set123.IsSubsetOf(set12));
  assert(set123.IsSubsetOf(set1234));
  assert(!set1234.IsSubsetOf(set12));
  assert(!set1234.IsSubsetOf(set123));

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestHashSet-TestIsSubsetOf", TestIsSubsetOf)

static int
TestUnionWith()
{
  using namespace jlm::util;

  HashSet<int> set12({ 1, 2 });
  HashSet<int> set123({ 1, 2, 3 });
  HashSet<int> set45({ 4, 5 });

  // Unioning with a subset should not change anything
  bool result = set123.UnionWith(set12);
  assert(!result);
  assert(set123.Size() == 3);

  // Putting {1, 2, 3} into {1, 2} should make it grow
  result = set12.UnionWith(set123);
  assert(result);
  assert(set12.Size() == 3);
  assert(set12 == set123);

  // Unioning again does nothing
  result = set12.UnionWith(set123);
  assert(!result);

  // Test union and clear
  result = set45.UnionWithAndClear(set123);
  assert(result);
  assert(set45.Size() == 5);
  assert(set123.IsEmpty());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestHashSet-TestUnionWith", TestUnionWith)

static int
TestIntersectWith()
{
  using namespace jlm::util;

  HashSet<int> set12({ 1, 2 });
  HashSet<int> set123({ 1, 2, 3 });
  HashSet<int> set45({ 4, 5 });

  set123.IntersectWith(set12);
  assert(set123 == set12);

  set123.IntersectWith(set45);
  assert(set123.Size() == 0);

  set123.Insert(1);
  set123.Insert(2);
  set123.Insert(3);
  set123.IntersectWithAndClear(set12);

  assert(set123.Size() == 2);
  assert(set12.Size() == 0);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestHashSet-TestIntersectWith", TestIntersectWith)

static int
TestDifferenceWith()
{
  using namespace jlm::util;

  HashSet<int> set12({ 1, 2 });
  HashSet<int> set123({ 1, 2, 3 });
  HashSet<int> set45({ 4, 5 });

  set123.DifferenceWith(set12); // {1, 2, 3} - {1, 2}
  assert(set123.Size() == 1);
  assert(set123.Contains(3));

  // set12 was not touched
  assert(set12.Size() == 2);

  // Create the set {0, 1, 3}
  set123.Insert(0);
  set123.Insert(1);

  set12.DifferenceWith(set123); // {1, 2} - {0, 1, 3}
  assert(set12.Size() == 1);
  assert(set12.Contains(2));

  // Difference with other sets becomes empty
  set45.DifferenceWith(set123);
  set45.DifferenceWith(set12);
  assert(set45.Size() == 2);

  // We handle the case where both sets are the same set without crashing
  set45.DifferenceWith(set45);
  assert(set45.IsEmpty());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestHashSet-TestDifferenceWith", TestDifferenceWith)

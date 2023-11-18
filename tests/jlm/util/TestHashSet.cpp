/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/HashSet.hpp>

#include <cassert>
#include <memory>

static void
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
}

static void
TestUniquePointer()
{
  jlm::util::HashSet<std::unique_ptr<int>> hashSet;

  hashSet.Insert(std::make_unique<int>(0));
  assert(hashSet.Size() == 1);

  hashSet.Remove(std::make_unique<int>(0));
  assert(hashSet.Size() == 1);

  hashSet.Clear();
  assert(hashSet.Size() == 0);
}

static void
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
}

static void
TestUnionWith()
{
  using namespace jlm::util;

  HashSet<int> set12({ 1, 2 });
  HashSet<int> set123({ 1, 2, 3 });
  HashSet<int> set45({ 4, 5 });

  assert(!set123.UnionWith(set12));

  assert(set12.UnionWith(set123));
  assert(!set12.UnionWith(set123));

  assert(set12.Size() == 3);
  assert(set12 == set123);

  assert(set45.UnionWith(set123));
  assert(set45.Size() == 5);
}

static void
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
}

static int
TestHashSet()
{
  TestInt();
  TestUniquePointer();
  TestIsSubsetOf();
  TestUnionWith();
  TestIntersectWith();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestHashSet", TestHashSet)

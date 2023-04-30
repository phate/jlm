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
  jlm::HashSet<int> hashSet({0, 1, 2, 3, 4, 5, 6, 7});

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
  assert(sum == (0+2+3+4+5+6+7+8));

  jlm::HashSet<int> hashSet2({8, 9});
  hashSet.UnionWith(hashSet2);
  assert(hashSet.Size() == 9);

  auto numRemoved = hashSet.RemoveWhere([](int n) -> bool { return n % 2 != 0; });
  assert(numRemoved == 4);

  hashSet.Clear();
  assert(hashSet.Size() == 0);
}

static void
TestUniquePointer()
{
  jlm::HashSet<std::unique_ptr<int>> hashSet;

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
  jlm::HashSet<int> set12({1, 2});
  jlm::HashSet<int> set123({1, 2, 3});
  jlm::HashSet<int> set1234({1, 2, 3, 4});

  assert(set12.IsSubsetOf(set12));
  assert(set12.IsSubsetOf(set123));
  assert(set12.IsSubsetOf(set1234));
  assert(!set123.IsSubsetOf(set12));
  assert(set123.IsSubsetOf(set1234));
  assert(!set1234.IsSubsetOf(set12));
  assert(!set1234.IsSubsetOf(set123));
}

static int
TestHashSet()
{
  TestInt();
  TestUniquePointer();
  TestIsSubsetOf();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestHashSet", TestHashSet)
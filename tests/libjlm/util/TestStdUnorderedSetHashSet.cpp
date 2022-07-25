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
  jlm::StdUnorderedSetHashSet<int> hashSet({0, 1, 2, 3, 4, 5, 6, 7});

  assert(hashSet.Size());
  assert(hashSet.Contains(3));
  assert(!hashSet.Contains(8));

  assert(hashSet.Insert(8));
  assert(hashSet.Contains(8));
  assert(!hashSet.Insert(8));

  assert(hashSet.Remove(1));
  assert(!hashSet.Contains(1));
  assert(!hashSet.Remove(1));

  hashSet.Clear();
  assert(hashSet.Size() == 0);
}

static void
TestUniquePointer()
{
  jlm::StdUnorderedSetHashSet<std::unique_ptr<int>> hashSet;

  hashSet.Insert(std::make_unique<int>(0));
  assert(hashSet.Size() == 1);

  hashSet.Remove(std::make_unique<int>(0));
  assert(hashSet.Size() == 1);

  hashSet.Clear();
  assert(hashSet.Size() == 0);
}

static int
TestStdUnorderedSetHashSet()
{
  TestInt();
  TestUniquePointer();

  return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/util/TestStdUnorderedSetHashSet", TestStdUnorderedSetHashSet)
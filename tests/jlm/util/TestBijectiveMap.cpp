/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <jlm/util/BijectiveMap.hpp>

#include <cassert>
#include <string>

static void
TestBijectiveMapConstructors()
{
  using namespace jlm::util;

  const BijectiveMap<int, std::string> emptyBiMap;
  assert(emptyBiMap.Size() == 0);

  // Construct from iterator range
  std::unordered_map<int, std::string> unordMap = { { 10, "Ten" }, { 20, "Twenty" } };
  BijectiveMap<int, std::string> rangeBiMap(unordMap.begin(), unordMap.end());
  assert(rangeBiMap.Size() == 2);

  // Make the unordered map illegal
  unordMap[30] = "Twenty";
  JLM_ASSERT_THROWS(BijectiveMap<int, std::string> biMap(unordMap.begin(), unordMap.end()));

  // Construct from initializer list
  BijectiveMap<int, std::string> biMap = { { 10, "Ten" }, { 20, "Twenty" } };
  assert(biMap.Size() == 2);

  // Construct from illegal initalizer list
  JLM_ASSERT_THROWS(BijectiveMap<int, std::string>{ { 10, "Ten" }, { 10, "Twenty" } });

  // Copy ctor
  BijectiveMap biMap2(biMap);
  assert(biMap2 == biMap);

  biMap2.Insert(30, "Thirty");
  assert(biMap2.Size() == 3);
  assert(biMap2 != biMap);

  // Move ctor
  BijectiveMap biMap3(std::move(biMap));
  assert(biMap3.Size() == 2);
  // All of biMap's content has been stolen
  assert(biMap.Size() == 0);

  // copy assignment
  biMap2 = biMap3;
  assert(biMap2.Size() == 2 && biMap2 == biMap3);

  // move assignment
  biMap = std::move(biMap3);
  assert(biMap.Size() == 2 && biMap == biMap2);
  assert(biMap3.Size() == 0);

  assert(biMap.LookupKey(10) == "Ten");
  assert(biMap.LookupKey(20) == "Twenty");
  assert(biMap.LookupValue("Ten") == 10);
  assert(biMap.LookupValue("Twenty") == 20);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/util/TestBijectiveMap-TestBijectiveMapConstructors",
    TestBijectiveMapConstructors)

static void
TestBijectiveMapClear()
{
  using namespace jlm::util;

  BijectiveMap<int, int> squares;
  squares.Insert(5, 25);
  squares.Insert(6, 36);
  assert(squares.Size() == 2);
  squares.Clear();
  assert(squares.Size() == 0);
  squares.Insert(5, 25);
  assert(squares.Size() == 1);
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestBijectiveMap-TestBijectiveMapClear", TestBijectiveMapClear)

static void
TestBijectiveMapInsert()
{
  using namespace jlm::util;

  BijectiveMap<int, int> squares;
  bool inserted = squares.Insert(5, 25);
  assert(inserted && squares.Size() == 1 && squares.HasKey(5) && squares.HasValue(25));
  inserted = squares.Insert(5, 45);
  assert(!inserted && squares.Size() == 1 && !squares.HasValue(45));

  JLM_ASSERT_THROWS(squares.InsertOrThrow(6, 25));
  JLM_ASSERT_THROWS(squares.InsertOrThrow(5, 36));

  inserted = squares.Insert(6, 36);
  assert(inserted && squares.Size() == 2);
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestBijectiveMap-TestBijectiveMapInsert", TestBijectiveMapInsert)

static void
TestBijectiveMapInsertPairs()
{
  using namespace jlm::util;

  BijectiveMap<int, std::string> biMap, biMap2;

  biMap.Insert(4, "four");
  biMap.Insert(6, "six");
  biMap.Insert(8, "eight");

  biMap2.Insert(4, "fire");
  biMap2.Insert(100, "six");
  biMap2.Insert(9, "nine");
  biMap2.Insert(10, "ten");

  const auto inserted = biMap.InsertPairs(biMap2.begin(), biMap2.end());
  assert(inserted == 2);
  assert(biMap.Size() == 5);
  assert(!biMap.HasKey(100) && !biMap.HasValue("fire"));
  assert(biMap.LookupKey(9) == "nine");
  assert(biMap.LookupValue("ten") == 10);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/util/TestBijectiveMap-TestBijectiveMapInsertPairs",
    TestBijectiveMapInsertPairs)

static void
TestBijectiveMapLookup()
{
  using namespace jlm::util;
  BijectiveMap<int, std::string> biMap({ { 1, "one" } });
  biMap.Insert(2, "two");

  assert(biMap.LookupKey(1) == "one");
  assert(biMap.LookupKey(2) == "two");
  assert(biMap.LookupValue("one") == 1);
  assert(biMap.LookupValue("two") == 2);

  JLM_ASSERT_THROWS((void)biMap.LookupKey(3));
  JLM_ASSERT_THROWS((void)biMap.LookupValue("three"));
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestBijectiveMap-TestBijectiveMapLookup", TestBijectiveMapLookup)

static void
TestBijectiveMapIterators()
{
  using namespace jlm::util;
  BijectiveMap<int, std::string> biMap({
      { 1, "one" },
      { 2, "two" },
      { 3, "three" },
  });

  size_t count = 0, sum = 0;
  for (const auto & [key, value] : biMap)
  {
    count++;
    sum += key;
    switch (key)
    {
    case 1:
      assert(value == "one");
      break;
    case 2:
      assert(value == "two");
      break;
    case 3:
      assert(value == "three");
      break;
    default:
      assert(false && "unreachable");
    }
  }
  assert(count == 3 && sum == 6);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/util/TestBijectiveMap-TestBijectiveMapIterators",
    TestBijectiveMapIterators)

static void
TestBijectiveMapErase()
{
  using namespace jlm::util;
  BijectiveMap<int, std::string> biMap({
      { 0, "zero" },
      { 1, "one" },
      { 2, "two" },

  });

  // Erasing leaves all other iterators valid
  const auto first = biMap.begin();
  const auto second = std::next(first);
  const auto third = std::next(second);
  const auto fourth = std::next(third);

  assert(fourth == biMap.end());

  const auto afterErased = biMap.Erase(second);
  assert(afterErased == third);
  assert(afterErased == std::next(first));
  assert(biMap.Size() == 2);
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestBijectiveMap-TestBijectiveMapErase", TestBijectiveMapErase)

static void
TestBijectiveMapRemoves()
{
  using namespace jlm::util;
  BijectiveMap<int, std::string> biMap({ { 1, "one" }, { 2, "two" }, { 3, "three" } });
  bool removed = biMap.RemoveKey(1);
  assert(removed && !biMap.HasKey(1) && !biMap.HasValue("one") && biMap.Size() == 2);
  removed = biMap.RemoveValue("two");
  assert(removed && !biMap.HasKey(2) && !biMap.HasValue("two") && biMap.Size() == 1);

  removed = biMap.RemoveKey(6);
  assert(!removed);
  removed = biMap.RemoveValue("five");
  assert(!removed);
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestBijectiveMap-TestBijectiveMapRemoves", TestBijectiveMapRemoves)

static void
TestBijectiveMapRemoveWhere()
{
  using namespace jlm::util;
  BijectiveMap<int, std::string> biMap({ { 1, "one" },
                                         { 2, "two" },
                                         { 3, "three" },
                                         { 4, "four" },
                                         { 5, "five" },
                                         { 6, "six" },
                                         { 7, "seven" },
                                         { 8, "eight" } });

  const auto KVPredicate = [](int key, const std::string & value)
  {
    return key < 4 || value == "eight";
  };
  auto removed = biMap.RemoveWhere(KVPredicate);
  assert(removed == 4);
  assert(!biMap.HasKey(8) && !biMap.HasValue("three"));

  removed = biMap.RemoveWhere(KVPredicate);
  assert(removed == 0);

  // Removes 5 and 7
  removed = biMap.RemoveKeysWhere(
      [&](int key)
      {
        return key % 2 == 1;
      });
  assert(removed == 2);
  assert(!biMap.HasKey(5) && !biMap.HasValue("seven"));

  // Removes 5 and 7
  removed = biMap.RemoveValuesWhere(
      [&](const std::string & value)
      {
        return value.size() > 3;
      });
  assert(removed == 1);
  assert(!biMap.HasKey(4));
  assert(biMap.Size() == 1 && biMap.HasValue("six"));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/util/TestBijectiveMap-TestBijectiveMapRemoveWhere",
    TestBijectiveMapRemoveWhere)

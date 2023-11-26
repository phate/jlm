/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/BiMap.hpp>

#include <cassert>
#include <string>

#define SHOULD_THROW_ERROR(code)                       \
  do                                                   \
  {                                                    \
    try                                                \
    {                                                  \
      code;                                            \
      assert(false && #code " was supposed to throw"); \
    }                                                  \
    catch (jlm::util::error e)                         \
    {                                                  \
      (void)e;                                         \
    }                                                  \
  } while (false)

static void
TestBiMapConstructors()
{
  using namespace jlm::util;

  const BiMap<int, std::string> emptyBiMap;
  assert(emptyBiMap.Size() == 0);

  // Construct from unordered_map
  BiMap<int, std::string> biMap({ { 10, "Ten" }, { 20, "Twenty" } });
  assert(biMap.Size() == 2);

  // Copy ctor
  BiMap biMap2(biMap);
  assert(biMap2 == biMap);

  biMap2.Insert(30, "Thirty");
  assert(biMap2.Size() == 3);
  assert(biMap2 != biMap);

  // Move ctor
  BiMap biMap3(std::move(biMap));
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

static void
TestBiMapInsert()
{
  using namespace jlm::util;

  BiMap<int, int> squares;
  squares.Insert(5, 25);
  assert(squares.Size() == 1 && squares.HasKey(5) && squares.HasValue(25));

  SHOULD_THROW_ERROR(squares.Insert(6, 25));
  SHOULD_THROW_ERROR(squares.Insert(5, 36));

  squares.Insert(6, 36);
  assert(squares.Size() == 2);

  squares.Clear();
  squares.Insert(5, 25);
  assert(squares.Size() == 1);
}

static void
TestBiMapLookup()
{
  using namespace jlm::util;
  BiMap<int, std::string> biMap({ { 1, "one" } });
  biMap.Insert(2, "two");

  assert(biMap.LookupKey(1) == "one");
  assert(biMap.LookupKey(2) == "two");
  assert(biMap.LookupValue("one") == 1);
  assert(biMap.LookupValue("two") == 2);

  SHOULD_THROW_ERROR((void)biMap.LookupKey(3));
  SHOULD_THROW_ERROR((void)biMap.LookupValue("three"));
}

static void
TestBiMapRemoves()
{
  using namespace jlm::util;
  BiMap<int, std::string> biMap({ { 1, "one" }, { 2, "two" }, { 3, "three" } });
  bool removed = biMap.RemoveKey(1);
  assert(removed && !biMap.HasKey(1) && !biMap.HasValue("one") && biMap.Size() == 2);
  removed = biMap.RemoveValue("two");
  assert(removed && !biMap.HasKey(2) && !biMap.HasValue("two") && biMap.Size() == 1);

  removed = biMap.RemoveKey(6);
  assert(!removed);
  removed = biMap.RemoveValue("five");
  assert(!removed);
}

static void
TestBiMapIterators()
{
  using namespace jlm::util;
  BiMap<int, std::string> biMap({ { 1, "one" }, { 2, "two" }, { 3, "three" } });

  for (const auto & [key, value] : biMap)
  {
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

  for (const auto & [key, value] : biMap.GetForwardMap())
  {
    assert(biMap.LookupValue(value) == key);
    assert(biMap.LookupKey(key) == value);
  }

  for (const auto & [value, key] : biMap.GetReverseMap())
  {
    assert(biMap.LookupValue(value) == key);
    assert(biMap.LookupKey(key) == value);
  }
}

static int
TestBiMap()
{
  TestBiMapConstructors();
  TestBiMapInsert();
  TestBiMapLookup();
  TestBiMapRemoves();
  TestBiMapIterators();
  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestBiMap", TestBiMap)

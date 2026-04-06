/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/util/BijectiveMap.hpp>

#include <string>

TEST(BijectiveMapTests, TestBijectiveMapConstructors)
{
  using namespace jlm::util;

  const BijectiveMap<int, std::string> emptyBiMap;
  EXPECT_EQ(emptyBiMap.Size(), 0u);

  // Construct from iterator range
  std::unordered_map<int, std::string> unordMap = { { 10, "Ten" }, { 20, "Twenty" } };
  BijectiveMap<int, std::string> rangeBiMap(unordMap.begin(), unordMap.end());
  EXPECT_EQ(rangeBiMap.Size(), 2u);

  // Make the unordered map illegal
  unordMap[30] = "Twenty";
  EXPECT_THROW(
      (BijectiveMap<int, std::string>{ unordMap.begin(), unordMap.end() }),
      jlm::util::Error);

  // Construct from initializer list
  BijectiveMap<int, std::string> biMap = { { 10, "Ten" }, { 20, "Twenty" } };
  assert(biMap.Size() == 2);

  // Construct from illegal initalizer list
  EXPECT_THROW(
      (BijectiveMap<int, std::string>{ { 10, "Ten" }, { 10, "Twenty" } }),
      jlm::util::Error);

  // Copy ctor
  BijectiveMap biMap2(biMap);
  EXPECT_EQ(biMap2, biMap);

  biMap2.Insert(30, "Thirty");
  EXPECT_EQ(biMap2.Size(), 3u);
  EXPECT_NE(biMap2, biMap);

  // Move ctor
  BijectiveMap biMap3(std::move(biMap));
  EXPECT_EQ(biMap3.Size(), 2u);
  // All of biMap's content has been stolen
  EXPECT_EQ(biMap.Size(), 0u);

  // copy assignment
  biMap2 = biMap3;
  EXPECT_EQ(biMap2.Size(), 2u);
  EXPECT_EQ(biMap2, biMap3);

  // move assignment
  biMap = std::move(biMap3);
  EXPECT_EQ(biMap.Size(), 2u);
  EXPECT_EQ(biMap, biMap2);
  EXPECT_EQ(biMap3.Size(), 0u);

  EXPECT_EQ(biMap.LookupKey(10), "Ten");
  EXPECT_EQ(biMap.LookupKey(20), "Twenty");
  EXPECT_EQ(biMap.LookupValue("Ten"), 10);
  EXPECT_EQ(biMap.LookupValue("Twenty"), 20);
}

TEST(BijectiveMapTests, TestBijectiveMapClear)
{
  using namespace jlm::util;

  BijectiveMap<int, int> squares;
  squares.Insert(5, 25);
  squares.Insert(6, 36);
  EXPECT_EQ(squares.Size(), 2u);
  squares.Clear();
  EXPECT_EQ(squares.Size(), 0u);
  squares.Insert(5, 25);
  EXPECT_EQ(squares.Size(), 1u);
}

TEST(BijectiveMapTests, TestBijectiveMapInsert)
{
  using namespace jlm::util;

  BijectiveMap<int, int> squares;
  bool inserted = squares.Insert(5, 25);
  EXPECT_EQ(inserted && squares.Size(), 1u);
  EXPECT_TRUE(squares.HasKey(5));
  EXPECT_TRUE(squares.HasValue(25));
  inserted = squares.Insert(5, 45);
  EXPECT_FALSE(inserted);
  EXPECT_EQ(squares.Size(), 1u);
  EXPECT_FALSE(squares.HasValue(45));

  EXPECT_THROW(squares.InsertOrThrow(6, 25), jlm::util::Error);
  EXPECT_THROW(squares.InsertOrThrow(5, 36), jlm::util::Error);

  inserted = squares.Insert(6, 36);
  EXPECT_TRUE(inserted);
  EXPECT_EQ(squares.Size(), 2u);
}

TEST(BijectiveMapTests, TestBijectiveMapInsertPairs)
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
  EXPECT_EQ(inserted, 2u);
  EXPECT_EQ(biMap.Size(), 5u);
  EXPECT_FALSE(biMap.HasKey(100));
  EXPECT_FALSE(biMap.HasValue("fire"));
  EXPECT_EQ(biMap.LookupKey(9), "nine");
  EXPECT_EQ(biMap.LookupValue("ten"), 10);
}

TEST(BijectiveMapTests, TestBijectiveMapLookup)
{
  using namespace jlm::util;
  BijectiveMap<int, std::string> biMap({ { 1, "one" } });
  biMap.Insert(2, "two");

  EXPECT_EQ(biMap.LookupKey(1), "one");
  EXPECT_EQ(biMap.LookupKey(2), "two");
  EXPECT_EQ(biMap.LookupValue("one"), 1);
  EXPECT_EQ(biMap.LookupValue("two"), 2);

  EXPECT_THROW((void)biMap.LookupKey(3), jlm::util::Error);
  EXPECT_THROW((void)biMap.LookupValue("three"), jlm::util::Error);
}

TEST(BijectiveMapTests, TestBijectiveMapIterators)
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
      EXPECT_EQ(value, "one");
      break;
    case 2:
      EXPECT_EQ(value, "two");
      break;
    case 3:
      EXPECT_EQ(value, "three");
      break;
    default:
      FAIL() << "unreachable";
    }
  }
  EXPECT_EQ(count, 3u);
  EXPECT_EQ(sum, 6u);
}

TEST(BijectiveMapTests, TestBijectiveMapErase)
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
  EXPECT_EQ(afterErased, third);
  EXPECT_EQ(afterErased, std::next(first));
  EXPECT_EQ(biMap.Size(), 2u);
}

TEST(BijectiveMapTests, TestBijectiveMapRemoves)
{
  using namespace jlm::util;
  BijectiveMap<int, std::string> biMap({ { 1, "one" }, { 2, "two" }, { 3, "three" } });
  bool removed = biMap.RemoveKey(1);
  EXPECT_TRUE(removed);
  EXPECT_FALSE(biMap.HasKey(1));
  EXPECT_FALSE(biMap.HasValue("one"));
  EXPECT_EQ(biMap.Size(), 2u);
  removed = biMap.RemoveValue("two");
  EXPECT_TRUE(removed);
  EXPECT_FALSE(biMap.HasKey(2));
  EXPECT_FALSE(biMap.HasValue("two"));
  EXPECT_EQ(biMap.Size(), 1u);

  removed = biMap.RemoveKey(6);
  EXPECT_FALSE(removed);
  removed = biMap.RemoveValue("five");
  EXPECT_FALSE(removed);
}

TEST(BijectiveMapTests, TestBijectiveMapRemoveWhere)
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
  EXPECT_EQ(removed, 4u);
  EXPECT_FALSE(biMap.HasKey(8));
  EXPECT_FALSE(biMap.HasValue("three"));

  removed = biMap.RemoveWhere(KVPredicate);
  EXPECT_EQ(removed, 0u);

  // Removes 5 and 7
  removed = biMap.RemoveKeysWhere(
      [&](int key)
      {
        return key % 2 == 1;
      });
  EXPECT_EQ(removed, 2u);
  EXPECT_FALSE(biMap.HasKey(5));
  EXPECT_FALSE(biMap.HasValue("seven"));

  // Removes 5 and 7
  removed = biMap.RemoveValuesWhere(
      [&](const std::string & value)
      {
        return value.size() > 3;
      });
  EXPECT_EQ(removed, 1u);
  EXPECT_TRUE(!biMap.HasKey(4));
  EXPECT_EQ(biMap.Size(), 1u);
  EXPECT_TRUE(biMap.HasValue("six"));
}

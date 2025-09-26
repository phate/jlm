/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/util/intrusive-hash.hpp>

#include <cassert>

struct MyItem
{
  MyItem(int k, int v)
      : key(k),
        value(v)
  {}

  int key;
  int value;

  struct
  {
    MyItem * prev;
    MyItem * next;
  } hash_chain{ nullptr, nullptr };
};

struct MyAccessor
{
  int
  get_key(const MyItem * item) const noexcept
  {
    return item->key;
  }

  MyItem *
  get_prev(const MyItem * item) const noexcept
  {
    return item->hash_chain.prev;
  }

  MyItem *
  get_next(const MyItem * item) const noexcept
  {
    return item->hash_chain.next;
  }

  void
  set_prev(MyItem * item, MyItem * prev) const noexcept
  {
    item->hash_chain.prev = prev;
  }

  void
  set_next(MyItem * item, MyItem * next) const noexcept
  {
    item->hash_chain.next = next;
  }
};

typedef jlm::util::IntrusiveHash<int, MyItem, MyAccessor> my_hash;

struct MyStringItem
{
  MyStringItem(const std::string & k, const std::string & v)
      : key(k),
        value(v)
  {}

  std::string key{};
  std::string value{};
  jlm::util::IntrusiveHashAnchor<MyStringItem> hash_chain{};

  typedef jlm::util::IntrusiveHashAccessor<
      std::string,
      MyStringItem,
      &MyStringItem::key,
      &MyStringItem::hash_chain>
      hash_accessor;
};

typedef jlm::util::IntrusiveHash<std::string, MyStringItem, MyStringItem::hash_accessor> my_strhash;

static void
test_int_hash()
{
  my_hash m;

  assert(m.find(42) == m.end());

  MyItem i1 = { 42, 0 };
  m.insert(&i1);
  assert(&*m.find(42) == &i1);

  MyItem i2 = { 10, 0 };
  m.insert(&i2);

  m.erase(&i1);
  assert(m.find(42) == m.end());
  m.insert(&i1);
  assert(&*m.find(42) == &i1);

  int seen_i1 = 0, seen_i2 = 0;
  for (const MyItem & i : m)
  {
    assert((&i == &i1) || (&i == &i2));
    if (&i == &i1)
    {
      ++seen_i1;
    }
    if (&i == &i2)
    {
      ++seen_i2;
    }
  }
  assert(seen_i1 == 1);
  assert(seen_i2 == 1);
}

static void
test_str_hash()
{
  my_strhash m;

  assert(m.find("42") == m.end());

  MyStringItem i1 = { "42", "0" };
  m.insert(&i1);
  assert(&*m.find("42") == &i1);

  MyStringItem i2 = { "10", "0" };
  m.insert(&i2);

  m.erase(&i1);
  assert(m.find("42") == m.end());
  m.insert(&i1);
  assert(&*m.find("42") == &i1);

  int seen_i1 = 0, seen_i2 = 0;
  for (const MyStringItem & i : m)
  {
    assert((&i == &i1) || (&i == &i2));
    if (&i == &i1)
    {
      ++seen_i1;
    }
    if (&i == &i2)
    {
      ++seen_i2;
    }
  }
  assert(seen_i1 == 1);
  assert(seen_i2 == 1);
}

static void
test_main()
{
  test_int_hash();
  test_str_hash();
}

JLM_UNIT_TEST_REGISTER("jlm/util/test-intrusive-hash", test_main)

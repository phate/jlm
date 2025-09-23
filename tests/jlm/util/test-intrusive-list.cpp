/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/util/intrusive-list.hpp>

#include <cassert>

namespace
{

struct MyItem
{
  MyItem()
      : p(nullptr)
  {}

  explicit MyItem(int * ptr)
      : p(ptr)
  {}

  ~MyItem()
  {
    if (p)
    {
      *p = 0;
    }
  }

  int * p;
  jlm::util::IntrusiveListAnchor<MyItem> anchor{};
  typedef jlm::util::IntrusiveListAccessor<MyItem, &MyItem::anchor> accessor;
};

typedef jlm::util::IntrusiveList<MyItem, MyItem::accessor> my_list;
typedef jlm::util::OwnerIntrusiveList<MyItem, MyItem::accessor> my_owner_list;

static void
test_simple_list()
{
  my_list l;

  assert(l.empty());
  MyItem i1, i2, i3;

  l.push_back(&i2);
  assert(l.begin().ptr() == &i2);
  assert(std::next(l.begin()) == l.end());
  assert(std::prev(l.end()).ptr() == &i2);

  l.insert(l.begin(), &i1);
  assert(l.begin().ptr() == &i1);
  assert(std::next(l.begin()).ptr() == &i2);
  assert(std::next(l.begin(), 2) == l.end());
  assert(std::prev(l.end()).ptr() == &i2);

  l.insert(l.end(), &i3);
  assert(l.begin().ptr() == &i1);
  assert(std::next(l.begin()).ptr() == &i2);
  assert(std::next(l.begin(), 2).ptr() == &i3);
  assert(std::next(l.begin(), 3) == l.end());
  assert(std::prev(l.end()).ptr() == &i3);
  assert(std::prev(l.end(), 2).ptr() == &i2);
  assert(std::prev(l.end(), 3).ptr() == &i1);

  l.erase(&i2);
  assert(l.begin().ptr() == &i1);
  assert(std::next(l.begin()).ptr() == &i3);
  assert(std::next(l.begin(), 2) == l.end());
  assert(std::prev(l.end()).ptr() == &i3);
  assert(std::prev(l.end(), 2).ptr() == &i1);

  my_list l2;
  l2.splice(l2.begin(), l);
  assert(l.empty());
  assert(l2.size() == 2);
}

static void
test_owner_list()
{
  int v1 = 1;
  int v2 = 2;
  int v3 = 3;

  {
    my_owner_list l;

    assert(l.empty());

    l.push_back(std::unique_ptr<MyItem>(new MyItem(&v2)));
    assert(l.begin()->p == &v2);
    assert(std::next(l.begin()) == l.end());
    assert(std::prev(l.end())->p == &v2);

    l.insert(l.begin(), std::unique_ptr<MyItem>(new MyItem(&v1)));
    assert(l.begin()->p == &v1);
    assert(std::next(l.begin())->p == &v2);
    assert(std::next(l.begin(), 2) == l.end());
    assert(std::prev(l.end())->p == &v2);

    l.insert(l.end(), std::unique_ptr<MyItem>(new MyItem(&v3)));
    assert(l.begin()->p == &v1);
    assert(std::next(l.begin())->p == &v2);
    assert(std::next(l.begin(), 2)->p == &v3);
    assert(std::next(l.begin(), 3) == l.end());
    assert(std::prev(l.end())->p == &v3);
    assert(std::prev(l.end(), 2)->p == &v2);
    assert(std::prev(l.end(), 3)->p == &v1);

    l.erase(std::next(l.begin()));
    assert(v1 == 1);
    assert(v2 == 0); // destructor should have been called
    assert(v3 == 3);

    assert(l.begin()->p == &v1);
    assert(std::next(l.begin())->p == &v3);
    assert(std::next(l.begin(), 2) == l.end());
    assert(std::prev(l.end())->p == &v3);
    assert(std::prev(l.end(), 2)->p == &v1);

    std::unique_ptr<MyItem> i = l.unlink(l.begin());
    assert(v1 == 1); // destructor not invoked, transferred ownership
    assert(v2 == 0);
    assert(v3 == 3);

    i.reset();
    assert(v1 == 0); // destructor called now

    assert(l.begin()->p == &v3);
    assert(std::next(l.begin()) == l.end());
    assert(std::prev(l.end())->p == &v3);

    my_owner_list l2;
    l2.splice(l2.begin(), l);
    assert(l.size() == 0);
    assert(l2.size() == 1);
  }
  assert(v3 == 0);
}

void
test_main()
{
  test_simple_list();
  test_owner_list();
}

}

JLM_UNIT_TEST_REGISTER("jlm/util/test-intrusive-list", test_main)

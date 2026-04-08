/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/util/intrusive-list.hpp>

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

TEST(IntrusiveListTests, test_simple_list)
{
  my_list l;

  EXPECT_TRUE(l.empty());
  MyItem i1, i2, i3;

  l.push_back(&i2);
  EXPECT_EQ(l.begin().ptr(), &i2);
  EXPECT_EQ(std::next(l.begin()), l.end());
  EXPECT_EQ(std::prev(l.end()).ptr(), &i2);

  l.insert(l.begin(), &i1);
  EXPECT_EQ(l.begin().ptr(), &i1);
  EXPECT_EQ(std::next(l.begin()).ptr(), &i2);
  EXPECT_EQ(std::next(l.begin(), 2), l.end());
  EXPECT_EQ(std::prev(l.end()).ptr(), &i2);

  l.insert(l.end(), &i3);
  EXPECT_EQ(l.begin().ptr(), &i1);
  EXPECT_EQ(std::next(l.begin()).ptr(), &i2);
  EXPECT_EQ(std::next(l.begin(), 2).ptr(), &i3);
  EXPECT_EQ(std::next(l.begin(), 3), l.end());
  EXPECT_EQ(std::prev(l.end()).ptr(), &i3);
  EXPECT_EQ(std::prev(l.end(), 2).ptr(), &i2);
  EXPECT_EQ(std::prev(l.end(), 3).ptr(), &i1);

  l.erase(&i2);
  EXPECT_EQ(l.begin().ptr(), &i1);
  EXPECT_EQ(std::next(l.begin()).ptr(), &i3);
  EXPECT_EQ(std::next(l.begin(), 2), l.end());
  EXPECT_EQ(std::prev(l.end()).ptr(), &i3);
  EXPECT_EQ(std::prev(l.end(), 2).ptr(), &i1);

  my_list l2;
  l2.splice(l2.begin(), l);
  EXPECT_TRUE(l.empty());
  EXPECT_EQ(l2.size(), 2u);
}

TEST(IntrusiveListTests, test_owner_list)
{
  int v1 = 1;
  int v2 = 2;
  int v3 = 3;

  {
    my_owner_list l;

    EXPECT_TRUE(l.empty());

    l.push_back(std::unique_ptr<MyItem>(new MyItem(&v2)));
    EXPECT_EQ(l.begin()->p, &v2);
    EXPECT_EQ(std::next(l.begin()), l.end());
    EXPECT_EQ(std::prev(l.end())->p, &v2);

    l.insert(l.begin(), std::unique_ptr<MyItem>(new MyItem(&v1)));
    EXPECT_EQ(l.begin()->p, &v1);
    EXPECT_EQ(std::next(l.begin())->p, &v2);
    EXPECT_EQ(std::next(l.begin(), 2), l.end());
    EXPECT_EQ(std::prev(l.end())->p, &v2);

    l.insert(l.end(), std::unique_ptr<MyItem>(new MyItem(&v3)));
    EXPECT_EQ(l.begin()->p, &v1);
    EXPECT_EQ(std::next(l.begin())->p, &v2);
    EXPECT_EQ(std::next(l.begin(), 2)->p, &v3);
    EXPECT_EQ(std::next(l.begin(), 3), l.end());
    EXPECT_EQ(std::prev(l.end())->p, &v3);
    EXPECT_EQ(std::prev(l.end(), 2)->p, &v2);
    EXPECT_EQ(std::prev(l.end(), 3)->p, &v1);

    l.erase(std::next(l.begin()));
    EXPECT_EQ(v1, 1);
    EXPECT_EQ(v2, 0); // destructor should have been called
    EXPECT_EQ(v3, 3);

    EXPECT_EQ(l.begin()->p, &v1);
    EXPECT_EQ(std::next(l.begin())->p, &v3);
    EXPECT_EQ(std::next(l.begin(), 2), l.end());
    EXPECT_EQ(std::prev(l.end())->p, &v3);
    EXPECT_EQ(std::prev(l.end(), 2)->p, &v1);

    std::unique_ptr<MyItem> i = l.unlink(l.begin());
    EXPECT_EQ(v1, 1); // destructor not invoked, transferred ownership
    EXPECT_EQ(v2, 0);
    EXPECT_EQ(v3, 3);

    i.reset();
    EXPECT_EQ(v1, 0); // destructor called now

    EXPECT_EQ(l.begin()->p, &v3);
    EXPECT_EQ(std::next(l.begin()), l.end());
    EXPECT_EQ(std::prev(l.end())->p, &v3);

    my_owner_list l2;
    l2.splice(l2.begin(), l);
    EXPECT_EQ(l.size(), 0u);
    EXPECT_EQ(l2.size(), 1u);
  }
  EXPECT_EQ(v3, 0);
}

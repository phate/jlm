/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/util/Worklist.hpp>

TEST(WorklistTests, TestLifoWorklist)
{
  jlm::util::LifoWorklist<size_t> wl;
  EXPECT_FALSE(wl.HasMoreWorkItems());
  wl.PushWorkItem(5);
  wl.PushWorkItem(2);
  wl.PushWorkItem(5);

  EXPECT_TRUE(wl.HasMoreWorkItems());
  auto item = wl.PopWorkItem();
  EXPECT_EQ(item, 2u);

  wl.PushWorkItem(7);
  item = wl.PopWorkItem();
  EXPECT_EQ(item, 7u);

  item = wl.PopWorkItem();
  EXPECT_EQ(item, 5u);
  EXPECT_FALSE(wl.HasMoreWorkItems());
}

TEST(WorklistTests, TestFifoWorklist)
{
  jlm::util::FifoWorklist<size_t> wl;
  EXPECT_FALSE(wl.HasMoreWorkItems());
  wl.PushWorkItem(5);
  wl.PushWorkItem(2);
  wl.PushWorkItem(5);

  EXPECT_TRUE(wl.HasMoreWorkItems());
  auto item = wl.PopWorkItem();
  EXPECT_EQ(item, 5u);

  wl.PushWorkItem(7);

  item = wl.PopWorkItem();
  EXPECT_EQ(item, 2u);

  item = wl.PopWorkItem();
  EXPECT_EQ(item, 7u);
  EXPECT_FALSE(wl.HasMoreWorkItems());
}

TEST(WorklistTests, TestLrfWorklist)
{
  jlm::util::LrfWorklist<size_t> wl;
  EXPECT_FALSE(wl.HasMoreWorkItems());
  wl.PushWorkItem(5);

  EXPECT_TRUE(wl.HasMoreWorkItems());
  auto item = wl.PopWorkItem();
  EXPECT_EQ(item, 5u);

  wl.PushWorkItem(7);
  wl.PushWorkItem(5);

  item = wl.PopWorkItem();
  EXPECT_EQ(item, 7u);

  wl.PushWorkItem(2);
  item = wl.PopWorkItem();
  EXPECT_EQ(item, 2u);

  item = wl.PopWorkItem();
  EXPECT_EQ(item, 5u);
  EXPECT_FALSE(wl.HasMoreWorkItems());
}

TEST(WorklistTests, TestTwoPhaseLrfWorklist)
{
  jlm::util::TwoPhaseLrfWorklist<size_t> wl;
  EXPECT_FALSE(wl.HasMoreWorkItems());
  wl.PushWorkItem(5);

  EXPECT_TRUE(wl.HasMoreWorkItems());
  auto item = wl.PopWorkItem();
  EXPECT_EQ(item, 5u);

  // These items are both pushed to next
  wl.PushWorkItem(7);
  wl.PushWorkItem(5);

  // Popping moves both items from next to current, and 7 has been fired least recently (never)
  item = wl.PopWorkItem();
  EXPECT_EQ(item, 7u);

  // Pushing 2 goes to next
  wl.PushWorkItem(2);
  // We still pop 5, since 5 is on the current list, despite 2 being less recently fired
  item = wl.PopWorkItem();
  EXPECT_EQ(item, 5u);

  item = wl.PopWorkItem();
  EXPECT_EQ(item, 2u);
  EXPECT_FALSE(wl.HasMoreWorkItems());
}

TEST(WorklistTests, TestWorkset)
{
  jlm::util::Workset<size_t> ws;
  EXPECT_FALSE(ws.HasMoreWorkItems());
  EXPECT_FALSE(ws.HasWorkItem(7));
  ws.PushWorkItem(7);
  EXPECT_TRUE(ws.HasMoreWorkItems());
  EXPECT_TRUE(ws.HasWorkItem(7));
  ws.PushWorkItem(5);
  EXPECT_TRUE(ws.HasWorkItem(5));
  EXPECT_TRUE(ws.HasWorkItem(7));
  ws.RemoveWorkItem(7);
  EXPECT_FALSE(ws.HasWorkItem(7));
  EXPECT_TRUE(ws.HasWorkItem(5));
  ws.RemoveWorkItem(5);
  EXPECT_FALSE(ws.HasMoreWorkItems());
}

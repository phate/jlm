/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/util/time.hpp>

#include <chrono>
#include <thread>

using namespace jlm::util;

static void
sleepUs(int us)
{
  std::this_thread::sleep_for(std::chrono::duration<int, std::micro>(us));
}

TEST(TimerTests, TestStartStop)
{
  Timer t;
  EXPECT_EQ(t.ns(), 0);
  EXPECT_FALSE(t.isRunning());

  t.start();
  EXPECT_TRUE(t.isRunning());
  sleepUs(10);
  t.stop();
  EXPECT_FALSE(t.isRunning());
  auto ns = t.ns();
  EXPECT_GE(ns, 10000);

  // Add more time
  t.start();
  sleepUs(1);
  t.stop();
  EXPECT_GE(t.ns(), ns + 1000);
}

TEST(TimerTests, TestReset)
{
  Timer t;
  EXPECT_EQ(t.ns(), 0);
  EXPECT_FALSE(t.isRunning());

  t.start();
  sleepUs(1);
  t.stop();

  EXPECT_NE(t.ns(), 0);
  t.reset();
  EXPECT_EQ(t.ns(), 0);
  EXPECT_FALSE(t.isRunning());

  // Resetting while running
  t.start();
  t.reset();
  EXPECT_EQ(t.ns(), 0);
  EXPECT_FALSE(t.isRunning());
}

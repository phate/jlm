/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/time.hpp>

#include <cassert>
#include <chrono>
#include <thread>

using namespace jlm::util;

static void
sleepUs(int us)
{
  std::this_thread::sleep_for(std::chrono::duration<int, std::micro>(us));
}

static void
TestStartStop()
{
  Timer t;
  assert(t.ns() == 0);
  assert(!t.isRunning());

  t.start();
  assert(t.isRunning());
  sleepUs(10);
  t.stop();
  assert(!t.isRunning());
  auto ns = t.ns();
  assert(ns >= 10000);

  // Add more time
  t.start();
  sleepUs(1);
  t.stop();
  assert(t.ns() >= ns + 1000);
}

static void
TestReset()
{
  Timer t;
  assert(t.ns() == 0);
  assert(!t.isRunning());

  t.start();
  sleepUs(1);
  t.stop();

  assert(t.ns() != 0);
  t.reset();
  assert(t.ns() == 0);
  assert(!t.isRunning());

  // Resetting while running
  t.start();
  t.reset();
  assert(t.ns() == 0);
  assert(!t.isRunning());
}

static void
TestTimer()
{
  TestStartStop();
  TestReset();
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestTimer", TestTimer)

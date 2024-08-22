/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include "tests/test-registry.hpp"

#include "jlm/util/Worklist.hpp"

#include <cassert>

static int
TestLifoWorklist()
{
  jlm::util::LifoWorklist<size_t> wl;
  assert(!wl.HasMoreWorkItems());
  wl.PushWorkItem(5);
  wl.PushWorkItem(2);
  wl.PushWorkItem(5);

  assert(wl.HasMoreWorkItems());
  auto item = wl.PopWorkItem();
  assert(item == 2);

  wl.PushWorkItem(7);
  item = wl.PopWorkItem();
  assert(item == 7);

  item = wl.PopWorkItem();
  assert(item == 5);
  assert(!wl.HasMoreWorkItems());

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestWorklist-TestLifoWorklist",
    TestLifoWorklist)

static int
TestFifoWorklist()
{
  jlm::util::FifoWorklist<size_t> wl;
  assert(!wl.HasMoreWorkItems());
  wl.PushWorkItem(5);
  wl.PushWorkItem(2);
  wl.PushWorkItem(5);

  assert(wl.HasMoreWorkItems());
  auto item = wl.PopWorkItem();
  assert(item == 5);

  wl.PushWorkItem(7);

  item = wl.PopWorkItem();
  assert(item == 2);

  item = wl.PopWorkItem();
  assert(item == 7);
  assert(!wl.HasMoreWorkItems());

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestWorklist-TestFifoWorklist",
    TestFifoWorklist)

static int
TestLrfWorklist()
{
  jlm::util::LrfWorklist<size_t> wl;
  assert(!wl.HasMoreWorkItems());
  wl.PushWorkItem(5);

  assert(wl.HasMoreWorkItems());
  auto item = wl.PopWorkItem();
  assert(item == 5);

  wl.PushWorkItem(7);
  wl.PushWorkItem(5);

  item = wl.PopWorkItem();
  assert(item == 7);

  wl.PushWorkItem(2);
  item = wl.PopWorkItem();
  assert(item == 2);

  item = wl.PopWorkItem();
  assert(item == 5);
  assert(!wl.HasMoreWorkItems());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestWorklist-TestLrfWorklist", TestLrfWorklist)

static int
TestTwoPhaseLrfWorklist()
{
  jlm::util::TwoPhaseLrfWorklist<size_t> wl;
  assert(!wl.HasMoreWorkItems());
  wl.PushWorkItem(5);

  assert(wl.HasMoreWorkItems());
  auto item = wl.PopWorkItem();
  assert(item == 5);

  // These items are both pushed to next
  wl.PushWorkItem(7);
  wl.PushWorkItem(5);

  // Popping moves both items from next to current, and 7 has been fired least recently (never)
  item = wl.PopWorkItem();
  assert(item == 7);

  // Pushing 2 goes to next
  wl.PushWorkItem(2);
  // We still pop 5, since 5 is on the current list, despite 2 being less recently fired
  item = wl.PopWorkItem();
  assert(item == 5);

  item = wl.PopWorkItem();
  assert(item == 2);
  assert(!wl.HasMoreWorkItems());

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestWorklist-TestTwoPhaseLrfWorklist",
    TestTwoPhaseLrfWorklist)

static int
TestDummyWorklist()
{
  jlm::util::DummyWorklist<size_t> wl;
  assert(!wl.HasPushBeenMade());
  wl.PushWorkItem(7);
  assert(wl.HasPushBeenMade());
  wl.ResetPush();
  assert(!wl.HasPushBeenMade());
  wl.ResetPush();
  assert(!wl.HasPushBeenMade());
  wl.PushWorkItem(7);
  assert(wl.HasPushBeenMade());

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestWorklist-TestDummyWorklist",
    TestDummyWorklist)

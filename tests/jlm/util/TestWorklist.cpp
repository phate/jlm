/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include "tests/test-registry.hpp"

#include "jlm/util/Worklist.hpp"

#include <cassert>

static void
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
}

static void
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
}

static void
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
}

static int
TestWorklist()
{
  TestLifoWorklist();
  TestFifoWorklist();
  TestLrfWorklist();
  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestWorklist", TestWorklist)

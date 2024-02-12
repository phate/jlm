/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>

#include <jlm/llvm/opt/alias-analyses/Worklist.hpp>

#include <cassert>

static void
TestLIFOWorklist()
{
  jlm::llvm::aa::LIFOWorklist<size_t> wl(10);
  assert(!wl.HasWorkItem());
  wl.PushWorkItem(5);
  wl.PushWorkItem(2);
  wl.PushWorkItem(5);

  assert(wl.HasWorkItem());
  auto item = wl.PopWorkItem();
  assert(item == 2);

  wl.PushWorkItem(7);
  item = wl.PopWorkItem();
  assert(item == 7);

  item = wl.PopWorkItem();
  assert(item == 5);
  assert(!wl.HasWorkItem());
}

static void
TestFIFOWorklist()
{
  jlm::llvm::aa::FIFOWorklist<size_t> wl(10);
  assert(!wl.HasWorkItem());
  wl.PushWorkItem(5);
  wl.PushWorkItem(2);
  wl.PushWorkItem(5);

  assert(wl.HasWorkItem());
  auto item = wl.PopWorkItem();
  assert(item == 5);

  wl.PushWorkItem(7);

  item = wl.PopWorkItem();
  assert(item == 2);

  item = wl.PopWorkItem();
  assert(item == 7);
  assert(!wl.HasWorkItem());
}

static void
TestLRFWorklist()
{
  jlm::llvm::aa::LRFWorklist<size_t> wl(10);
  assert(!wl.HasWorkItem());
  wl.PushWorkItem(5);

  assert(wl.HasWorkItem());
  auto item = wl.PopWorkItem();
  assert(item == 5);

  wl.PushWorkItem(7);
  wl.PushWorkItem(5);

  item = wl.PopWorkItem();
  assert(item == 7);

  wl.PushWorkItem(2);

  item = wl.PopWorkItem();
  assert(item == 5);

  item = wl.PopWorkItem();
  assert(item == 2);
  assert(!wl.HasWorkItem());
}

static int
TestWorklist()
{
  TestLIFOWorklist();
  TestFIFOWorklist();
  TestLRFWorklist();
  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestWorklist", TestWorklist)

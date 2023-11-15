/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>

#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>

#include <cassert>

static int
TestAndersen()
{
  return 0;
}

JLM_UNIT_TEST_REGISTER(
"jlm/llvm/opt/alias-analyses/TestAndersen",
TestAndersen)
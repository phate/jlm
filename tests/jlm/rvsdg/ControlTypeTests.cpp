/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/control.hpp>

#include <cassert>

static int
TestComputeHash()
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();

  ctltype controlType1(8);
  ctltype controlType2(16);
  ctltype controlType3(8);
  bittype bitType(8);

  // Act & Assert
  assert(controlType1.ComputeHash() == controlType1.ComputeHash());
  assert(controlType1.ComputeHash() != controlType2.ComputeHash());
  assert(controlType1.ComputeHash() == controlType3.ComputeHash());

  assert(controlType2.ComputeHash() == controlType2.ComputeHash());
  assert(controlType2.ComputeHash() != controlType3.ComputeHash());

  assert(controlType3.ComputeHash() == controlType3.ComputeHash());
  assert(controlType3.ComputeHash() != valueType->ComputeHash());
  assert(controlType3.ComputeHash() != bitType.ComputeHash());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/ControlTypeTests-TestComputeHash", TestComputeHash);

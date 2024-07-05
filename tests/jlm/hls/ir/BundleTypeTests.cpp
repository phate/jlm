/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/hls/ir/hls.hpp>

#include <cassert>

static int
TestComputeHash()
{
  using namespace jlm::hls;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto stateType = jlm::tests::statetype::Create();

  bundletype bundleType1({ { "foo", valueType } });
  bundletype bundleType2({ { "bar", stateType } });
  bundletype bundleType3({ { "foo", valueType } });

  // Act & Assert
  assert(bundleType1.ComputeHash() == bundleType1.ComputeHash());
  assert(bundleType1.ComputeHash() != bundleType2.ComputeHash());
  assert(bundleType1.ComputeHash() == bundleType3.ComputeHash());

  assert(bundleType2.ComputeHash() == bundleType2.ComputeHash());
  assert(bundleType2.ComputeHash() != bundleType3.ComputeHash());

  assert(bundleType3.ComputeHash() == bundleType3.ComputeHash());
  assert(bundleType3.ComputeHash() != valueType->ComputeHash());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/hls/ir/TriggerTypeTests-TestComputeHash", TestComputeHash);

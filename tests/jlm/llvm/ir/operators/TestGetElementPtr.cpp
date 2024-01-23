/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/ir/operators/GetElementPtr.hpp>

static void
TestOperationEquality()
{
  using namespace jlm::llvm;

  arraytype arrayType(jlm::rvsdg::bit8, 11);

  auto declaration1 = StructType::Declaration::Create({ &jlm::rvsdg::bit64, &jlm::rvsdg::bit64 });
  auto declaration2 = StructType::Declaration::Create({ &arrayType, &jlm::rvsdg::bit32 });

  StructType structType1(false, *declaration1);
  StructType structType2("myStructType", false, *declaration2);

  GetElementPtrOperation operation1({ jlm::rvsdg::bit32, jlm::rvsdg::bit32 }, structType1);
  GetElementPtrOperation operation2({ jlm::rvsdg::bit32, jlm::rvsdg::bit32 }, structType2);

  assert(operation1 != operation2);
}

static int
Test()
{
  TestOperationEquality();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/TestGetElementPtr", Test)

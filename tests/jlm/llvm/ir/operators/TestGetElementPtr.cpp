/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/ir/operators/GetElementPtr.hpp>

static void
TestOperationEquality()
{
  using namespace jlm;

  arraytype arrayType(jive::bit8, 11);

  auto declaration1 = jive::rcddeclaration::create({&jive::bit64, &jive::bit64});
  auto declaration2 = jive::rcddeclaration::create({&arrayType, &jive::bit32});

  StructType structType1(false, *declaration1);
  StructType structType2("myStructType", false, *declaration2);

  GetElementPtrOperation operation1({jive::bit32, jive::bit32}, structType1);
  GetElementPtrOperation operation2({jive::bit32, jive::bit32}, structType2);

  assert(operation1 != operation2);
}

static int
Test()
{
  TestOperationEquality();

  return 0;
}


JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/TestGetElementPtr", Test)

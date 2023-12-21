/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/ir/operators/operators.hpp>

static void
TestFreeConstructor()
{
  using namespace jlm::llvm;

  // Arrange and Act
  free_op free0(0);
  free_op free2(2);

  // Assert
  assert(free0.narguments() == 2);
  assert(free0.nresults() == 1);

  assert(free2.narguments() == 4);
  assert(free2.nresults() == 3);
}

static void
TestEqualityOperator()
{
  using namespace jlm::llvm;

  // Arrange
  free_op free0(0);
  free_op free1(1);
  free_op free2(1);

  // Act and Assert
  assert(free0 != free1);
  assert(free1 == free1);
  assert(free1 != free2); // 2 different instances should not compare equal
}

static void
TestThreeAddressCodeCreator()
{
  using namespace jlm::llvm;

  // Arrange
  ipgraph_module ipgModule(jlm::util::filepath(""), "", "");

  auto address = ipgModule.create_variable(PointerType(), "p");
  auto memoryState = ipgModule.create_variable(MemoryStateType(), "m");
  auto iOState = ipgModule.create_variable(iostatetype(), "io");

  // Act
  auto free0 = free_op::create(address, {}, iOState);
  auto free1 = free_op::create(address, { memoryState }, iOState);

  // Assert
  assert(free0->nresults() == 1);
  assert(free1->nresults() == 2);
}

static void
TestRvsdgCreator()
{
  using namespace jlm::llvm;

  // Arrange
  jlm::rvsdg::graph rvsdg;

  auto address = rvsdg.add_import({ PointerType(), "p" });
  auto memoryState = rvsdg.add_import({ MemoryStateType(), "m" });
  auto iOState = rvsdg.add_import({ iostatetype(), "io" });

  // Act
  auto freeResults0 = free_op::create(address, {}, iOState);
  auto freeResults1 = free_op::create(address, { memoryState }, iOState);

  // Assert
  assert(freeResults0.size() == 1);
  assert(freeResults1.size() == 2);
}

static int
TestFree()
{
  TestFreeConstructor();
  TestEqualityOperator();

  TestThreeAddressCodeCreator();
  TestRvsdgCreator();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/TestFree", TestFree)

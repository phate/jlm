/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>

#include <jlm/llvm/ir/operators/operators.hpp>

static void
TestFreeConstructor()
{
  using namespace jlm::llvm;

  // Arrange and Act
  FreeOperation free0(0);
  FreeOperation free2(2);

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
  FreeOperation free0(0);
  FreeOperation free1(1);
  FreeOperation free2(1);

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
  ipgraph_module ipgModule(jlm::util::FilePath(""), "", "");

  auto address = ipgModule.create_variable(PointerType::Create(), "p");
  auto memoryState = ipgModule.create_variable(MemoryStateType::Create(), "m");
  auto iOState = ipgModule.create_variable(IOStateType::Create(), "io");

  // Act
  auto free0 = FreeOperation::Create(address, {}, iOState);
  auto free1 = FreeOperation::Create(address, { memoryState }, iOState);

  // Assert
  assert(free0->nresults() == 1);
  assert(free1->nresults() == 2);
}

static void
TestRvsdgCreator()
{
  using namespace jlm::llvm;

  // Arrange
  jlm::rvsdg::Graph rvsdg;

  auto address = &jlm::tests::GraphImport::Create(rvsdg, PointerType::Create(), "p");
  auto memoryState = &jlm::tests::GraphImport::Create(rvsdg, MemoryStateType::Create(), "m");
  auto iOState = &jlm::tests::GraphImport::Create(rvsdg, IOStateType::Create(), "io");

  // Act
  auto freeResults0 = FreeOperation::Create(address, {}, iOState);
  auto freeResults1 = FreeOperation::Create(address, { memoryState }, iOState);

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

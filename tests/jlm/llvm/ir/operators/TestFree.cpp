/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/operators.hpp>

TEST(FreeOperationTests, TestFreeConstructor)
{
  using namespace jlm::llvm;

  // Arrange and Act
  FreeOperation free0(0);
  FreeOperation free2(2);

  // Assert
  EXPECT_EQ(free0.narguments(), 2);
  EXPECT_EQ(free0.nresults(), 1);

  EXPECT_EQ(free2.narguments(), 4);
  EXPECT_EQ(free2.nresults(), 3);
}

TEST(FreeOperationTests, TestEqualityOperator)
{
  using namespace jlm::llvm;

  // Arrange
  FreeOperation free0(0);
  FreeOperation free1(1);
  FreeOperation free2(1);

  // Act and Assert
  EXPECT_NE(free0, free1);
  EXPECT_EQ(free1, free1);
  EXPECT_NE(free1, free2); // 2 different instances should not compare equal
}

TEST(FreeOperationTests, TestThreeAddressCodeCreator)
{
  using namespace jlm::llvm;

  // Arrange
  InterProceduralGraphModule ipgModule(jlm::util::FilePath(""), "", "");

  auto address = ipgModule.create_variable(PointerType::Create(), "p");
  auto memoryState = ipgModule.create_variable(MemoryStateType::Create(), "m");
  auto iOState = ipgModule.create_variable(IOStateType::Create(), "io");

  // Act
  auto free0 = FreeOperation::Create(address, {}, iOState);
  auto free1 = FreeOperation::Create(address, { memoryState }, iOState);

  // Assert
  EXPECT_EQ(free0->nresults(), 1);
  EXPECT_EQ(free1->nresults(), 2);
}

TEST(FreeOperationTests, TestRvsdgCreator)
{
  using namespace jlm::llvm;

  // Arrange
  jlm::rvsdg::Graph rvsdg;

  auto address = &jlm::rvsdg::GraphImport::Create(rvsdg, PointerType::Create(), "p");
  auto memoryState = &jlm::rvsdg::GraphImport::Create(rvsdg, MemoryStateType::Create(), "m");
  auto iOState = &jlm::rvsdg::GraphImport::Create(rvsdg, IOStateType::Create(), "io");

  // Act
  auto freeResults0 = FreeOperation::Create(address, {}, iOState);
  auto freeResults1 = FreeOperation::Create(address, { memoryState }, iOState);

  auto & freeNode0 = jlm::rvsdg::AssertGetOwnerNode<jlm::rvsdg::SimpleNode>(*freeResults0[0]);

  // Assert
  EXPECT_EQ(freeResults0.size(), 1);
  EXPECT_EQ(freeResults1.size(), 2);
  EXPECT_EQ(FreeOperation::addressInput(freeNode0).origin(), address);
}

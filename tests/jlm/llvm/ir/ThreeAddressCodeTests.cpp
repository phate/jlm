/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>

TEST(ThreeAddressCodeTests, ToAscii)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();

  Variable v0(valueType, "v0");
  Variable v1(valueType, "v1");

  auto tac0 = ThreeAddressCode::create(TestOperation::create({}, {}), {});
  auto tac1 = ThreeAddressCode::create(TestOperation::create({ valueType }, {}), { &v0 });
  auto tac2 =
      ThreeAddressCode::create(TestOperation::create({ valueType, valueType }, {}), { &v0, &v1 });
  auto tac3 = ThreeAddressCode::create(TestOperation::create({}, { valueType }), {});
  auto tac4 = ThreeAddressCode::create(TestOperation::create({}, { valueType, valueType }), {});
  auto tac5 = ThreeAddressCode::create(
      TestOperation::create({ valueType, valueType }, { valueType, valueType }),
      { &v0, &v1 });

  // Act
  auto tac0String = ThreeAddressCode::ToAscii(*tac0);
  std::cout << tac0String << "\n" << std::flush;

  auto tac1String = ThreeAddressCode::ToAscii(*tac1);
  std::cout << tac1String << "\n" << std::flush;

  auto tac2String = ThreeAddressCode::ToAscii(*tac2);
  std::cout << tac2String << "\n" << std::flush;

  auto tac3String = ThreeAddressCode::ToAscii(*tac3);
  std::cout << tac3String << "\n" << std::flush;

  auto tac4String = ThreeAddressCode::ToAscii(*tac4);
  std::cout << tac4String << "\n" << std::flush;

  auto tac5String = ThreeAddressCode::ToAscii(*tac5);
  std::cout << tac5String << "\n" << std::flush;

  // Assert
  EXPECT_EQ(tac0String, "TestOperation");
  EXPECT_EQ(tac1String, "TestOperation v0");
  EXPECT_EQ(tac2String, "TestOperation v0, v1");
  // FIXME: We use a global static counter for the tv identifiers of the results. This means with
  // all the unit tests being merged into a single test executable that the counter is not
  // "predictable" any longer. The solution here is to get rid of this counter.
  // EXPECT_EQ(tac3String, "tv0 = TestOperation");
  // EXPECT_EQ(tac4String, "tv1, tv2 = TestOperation");
  // EXPECT_EQ(tac5String, "tv3, tv4 = TestOperation v0, v1");
}

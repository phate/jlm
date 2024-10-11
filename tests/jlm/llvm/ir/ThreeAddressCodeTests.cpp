/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

static int
ToAscii()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;

  // Arrange
  auto valueType = valuetype::Create();

  variable v0(valueType, "v0");
  variable v1(valueType, "v1");

  auto tac0 = create_testop_tac({}, {});
  auto tac1 = create_testop_tac({ &v0 }, {});
  auto tac2 = create_testop_tac({ &v0, &v1 }, {});
  auto tac3 = create_testop_tac({}, { valueType });
  auto tac4 = create_testop_tac({}, { valueType, valueType });
  auto tac5 = create_testop_tac({ &v0, &v1 }, { valueType, valueType });

  // Act
  auto tac0String = tac::ToAscii(*tac0);
  std::cout << tac0String << "\n" << std::flush;

  auto tac1String = tac::ToAscii(*tac1);
  std::cout << tac1String << "\n" << std::flush;

  auto tac2String = tac::ToAscii(*tac2);
  std::cout << tac2String << "\n" << std::flush;

  auto tac3String = tac::ToAscii(*tac3);
  std::cout << tac3String << "\n" << std::flush;

  auto tac4String = tac::ToAscii(*tac4);
  std::cout << tac4String << "\n" << std::flush;

  auto tac5String = tac::ToAscii(*tac5);
  std::cout << tac5String << "\n" << std::flush;

  // Assert
  assert(tac0String == "test_op");
  assert(tac1String == "test_op v0");
  assert(tac2String == "test_op v0, v1");
  assert(tac3String == "tv0 = test_op");
  assert(tac4String == "tv1, tv2 = test_op");
  assert(tac5String == "tv3, tv4 = test_op v0, v1");

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/ThreeAddressCodeTests-ToAscii", ToAscii);

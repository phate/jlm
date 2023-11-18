/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/Math.hpp>

#include <cassert>
#include <cstdint>

using namespace jlm::util;

static void
TestLog2Floor()
{
  auto testByteRange = [](auto type)
  {
    using T = decltype(type);
    assert(Log2Floor<T>(-10) == -1);
    assert(Log2Floor<T>(-1) == -1);
    assert(Log2Floor<T>(0) == -1);
    assert(Log2Floor<T>(1) == 0);
    assert(Log2Floor<T>(2) == 1);
    assert(Log2Floor<T>(3) == 1);
    assert(Log2Floor<T>(4) == 2);
    assert(Log2Floor<T>(7) == 2);
    assert(Log2Floor<T>(8) == 3);
    assert(Log2Floor<T>(63) == 5);
    assert(Log2Floor<T>(64) == 6);
    assert(Log2Floor<T>(127) == 6);
  };

  testByteRange(int8_t(0));
  testByteRange(int16_t(0));
  testByteRange(int32_t(0));
  testByteRange(int64_t(0));

  assert(Log2Floor<uint16_t>(0x7FFF) == 14);
  assert(Log2Floor<uint16_t>(0x8000) == 15);
  assert(Log2Floor<uint16_t>(0xFFFF) == 15);

  assert(Log2Floor<uint32_t>(0x7FFFFFFF) == 30);
  assert(Log2Floor<uint32_t>(0x80000000) == 31);
  assert(Log2Floor<uint32_t>(0xFFFFFFFF) == 31);
}

static void
TestBitsRequiredToRepresent()
{
  assert(BitsRequiredToRepresent<int8_t>(-1) == 8);
  assert(BitsRequiredToRepresent<int32_t>(-1) == 32);

  assert(BitsRequiredToRepresent<uint8_t>(0) == 0);
  assert(BitsRequiredToRepresent<uint8_t>(1) == 1);
  assert(BitsRequiredToRepresent<uint8_t>(2) == 2);
  assert(BitsRequiredToRepresent<uint8_t>(3) == 2);
  assert(BitsRequiredToRepresent<uint8_t>(4) == 3);
  assert(BitsRequiredToRepresent<uint8_t>(7) == 3);
  assert(BitsRequiredToRepresent<uint8_t>(8) == 4);

  assert(BitsRequiredToRepresent(0xFFFF) == 16);
  assert(BitsRequiredToRepresent(0x10000) == 17);
  assert(BitsRequiredToRepresent(0xFFFF'FFFF) == 32);
  assert(BitsRequiredToRepresent(0xFFFF'FFFFu) == 32);
  assert(BitsRequiredToRepresent(0xFFFF'FFFF'FFFF'FFFFll) == 64);
  assert(BitsRequiredToRepresent(0xFFFF'FFFF'FFFF'FFFFull) == 64);
}

static void
TestBitWidthOfEnum()
{
  enum class TestEnum1
  {
    Zero
  };

  enum class TestEnum4
  {
    Zero,
    One,
    Two,
    Three
  };

  enum class TestEnum5
  {
    Zero,
    One,
    Two,
    Three,
    Four
  };

  enum class TestEnum127
  {
    Zero = 0,
    OneHundredAndTwentySeven = 127
  };

  assert(BitWidthOfEnum(TestEnum1::Zero) == 0);
  assert(BitWidthOfEnum(TestEnum4::Three) == 2);
  assert(BitWidthOfEnum(TestEnum5::Four) == 3);
  assert(BitWidthOfEnum(TestEnum127::OneHundredAndTwentySeven) == 7);
}

static int
TestMath()
{
  TestLog2Floor();
  TestBitsRequiredToRepresent();
  TestBitWidthOfEnum();
  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestMath", TestMath)

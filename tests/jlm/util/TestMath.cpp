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

  assert(log2Floor<uint16_t>(0x7FFF) == 14);
  assert(log2Floor<uint16_t>(0x8000) == 15);
  assert(log2Floor<uint16_t>(0xFFFF) == 15);

  assert(log2Floor<uint32_t>(0x7FFFFFFF) == 30);
  assert(log2Floor<uint32_t>(0x80000000) == 31);
  assert(log2Floor<uint32_t>(0xFFFFFFFF) == 31);
}
JLM_UNIT_TEST_REGISTER("jlm/util/TestMath-TestLog2Floor", TestLog2Floor)

static void
TestRoundUpToPowerOf2()
{
  assert(RoundUpToPowerOf2<int32_t>(-10) == 1);
  assert(RoundUpToPowerOf2<int32_t>(0) == 1);
  assert(RoundUpToPowerOf2<uint32_t>(0) == 1);
  assert(RoundUpToPowerOf2<int32_t>(1) == 1);
  assert(RoundUpToPowerOf2<uint32_t>(1) == 1);
  assert(RoundUpToPowerOf2<uint32_t>(2) == 2);
  assert(RoundUpToPowerOf2<uint32_t>(3) == 4);

  assert(RoundUpToPowerOf2<uint32_t>(255) == 256);
  assert(RoundUpToPowerOf2<uint32_t>(256) == 256);
  assert(RoundUpToPowerOf2<uint32_t>(257) == 512);
  assert(RoundUpToPowerOf2<uint64_t>(0xFFFFFFFF) == 0x100000000ul);
}
JLM_UNIT_TEST_REGISTER("jlm/util/TestMath-TestRoundUpToPowerOf2", TestRoundUpToPowerOf2)

static void
TestRoundUpToMultipleOf()
{
  for (int i = -20; i <= 20; i++)
  {
    assert(RoundUpToMultipleOf<int32_t>(i, 1) == i);
  }

  assert(RoundUpToMultipleOf<int32_t>(0, 5) == 0);
  assert(RoundUpToMultipleOf<int32_t>(1, 5) == 5);
  assert(RoundUpToMultipleOf<int32_t>(4, 5) == 5);
  assert(RoundUpToMultipleOf<int32_t>(5, 5) == 5);
  assert(RoundUpToMultipleOf<int32_t>(6, 5) == 10);
  assert(RoundUpToMultipleOf<int32_t>(123, 5) == 125);
  assert(RoundUpToMultipleOf<int32_t>(8567, 2000) == 10'000);
  assert(RoundUpToMultipleOf<uint32_t>(8567, 2000) == 10'000);

  assert(RoundUpToMultipleOf<int32_t>(-1, 7) == 0);
  assert(RoundUpToMultipleOf<int32_t>(-6, 7) == 0);
  assert(RoundUpToMultipleOf<int32_t>(-7, 7) == -7);
  assert(RoundUpToMultipleOf<int32_t>(-8, 7) == -7);
  assert(RoundUpToMultipleOf<int32_t>(-14, 7) == -14);
  assert(RoundUpToMultipleOf<int32_t>(-15, 7) == -14);
  assert(RoundUpToMultipleOf<int32_t>(-14'006, 7) == -14'000);

  // Test different int sizes
  assert(RoundUpToMultipleOf<uint8_t>(13, 7) == 14);
  assert(RoundUpToMultipleOf<uint16_t>(13, 7) == 14);
  assert(RoundUpToMultipleOf<uint32_t>(13, 7) == 14);
  assert(RoundUpToMultipleOf<uint64_t>(13, 7) == 14);

  assert(RoundUpToMultipleOf<int8_t>(-13, 7) == -7);
  assert(RoundUpToMultipleOf<int16_t>(-13, 7) == -7);
  assert(RoundUpToMultipleOf<int32_t>(-13, 7) == -7);
  assert(RoundUpToMultipleOf<int64_t>(-13, 7) == -7);
}
JLM_UNIT_TEST_REGISTER("jlm/util/TestMath-TestRoundUpToMultipleOf", TestRoundUpToMultipleOf)

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
JLM_UNIT_TEST_REGISTER("jlm/util/TestMath-TestBitsRequiredToRepresent", TestBitsRequiredToRepresent)

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
JLM_UNIT_TEST_REGISTER("jlm/util/TestMath-TestBitWidthOfEnum", TestBitWidthOfEnum)

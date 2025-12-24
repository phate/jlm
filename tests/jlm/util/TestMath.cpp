/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/util/Math.hpp>

using namespace jlm::util;

TEST(MathTests, TestLog2Floor)
{
  auto testByteRange = [](auto type)
  {
    using T = decltype(type);
    EXPECT_EQ(log2Floor<T>(-10), -1);
    EXPECT_EQ(log2Floor<T>(-1), -1);
    EXPECT_EQ(log2Floor<T>(0), -1);
    EXPECT_EQ(log2Floor<T>(1), 0);
    EXPECT_EQ(log2Floor<T>(2), 1);
    EXPECT_EQ(log2Floor<T>(3), 1);
    EXPECT_EQ(log2Floor<T>(4), 2);
    EXPECT_EQ(log2Floor<T>(7), 2);
    EXPECT_EQ(log2Floor<T>(8), 3);
    EXPECT_EQ(log2Floor<T>(63), 5);
    EXPECT_EQ(log2Floor<T>(64), 6);
    EXPECT_EQ(log2Floor<T>(127), 6);
  };

  testByteRange(int8_t(0));
  testByteRange(int16_t(0));
  testByteRange(int32_t(0));
  testByteRange(int64_t(0));

  EXPECT_EQ(log2Floor<uint16_t>(0x7FFF), 14);
  EXPECT_EQ(log2Floor<uint16_t>(0x8000), 15);
  EXPECT_EQ(log2Floor<uint16_t>(0xFFFF), 15);

  EXPECT_EQ(log2Floor<uint32_t>(0x7FFFFFFF), 30);
  EXPECT_EQ(log2Floor<uint32_t>(0x80000000), 31);
  EXPECT_EQ(log2Floor<uint32_t>(0xFFFFFFFF), 31);
}

TEST(MathTests, TestRoundUpToPowerOf2)
{
  EXPECT_EQ(RoundUpToPowerOf2<int32_t>(-10), 1);
  EXPECT_EQ(RoundUpToPowerOf2<int32_t>(0), 1);
  EXPECT_EQ(RoundUpToPowerOf2<uint32_t>(0), 1);
  EXPECT_EQ(RoundUpToPowerOf2<int32_t>(1), 1);
  EXPECT_EQ(RoundUpToPowerOf2<uint32_t>(1), 1);
  EXPECT_EQ(RoundUpToPowerOf2<uint32_t>(2), 2);
  EXPECT_EQ(RoundUpToPowerOf2<uint32_t>(3), 4);

  EXPECT_EQ(RoundUpToPowerOf2<uint32_t>(255), 256);
  EXPECT_EQ(RoundUpToPowerOf2<uint32_t>(256), 256);
  EXPECT_EQ(RoundUpToPowerOf2<uint32_t>(257), 512);
  EXPECT_EQ(RoundUpToPowerOf2<uint64_t>(0xFFFFFFFF), 0x100000000ul);
}

TEST(MathTests, TestRoundUpToMultipleOf)
{
  for (int i = -20; i <= 20; i++)
  {
    EXPECT_EQ(RoundUpToMultipleOf<int32_t>(i, 1), i);
  }

  EXPECT_EQ(RoundUpToMultipleOf<int32_t>(0, 5), 0);
  EXPECT_EQ(RoundUpToMultipleOf<int32_t>(1, 5), 5);
  EXPECT_EQ(RoundUpToMultipleOf<int32_t>(4, 5), 5);
  EXPECT_EQ(RoundUpToMultipleOf<int32_t>(5, 5), 5);
  EXPECT_EQ(RoundUpToMultipleOf<int32_t>(6, 5), 10);
  EXPECT_EQ(RoundUpToMultipleOf<int32_t>(123, 5), 125);
  EXPECT_EQ(RoundUpToMultipleOf<int32_t>(8567, 2000), 10'000);
  EXPECT_EQ(RoundUpToMultipleOf<uint32_t>(8567, 2000), 10'000);

  EXPECT_EQ(RoundUpToMultipleOf<int32_t>(-1, 7), 0);
  EXPECT_EQ(RoundUpToMultipleOf<int32_t>(-6, 7), 0);
  EXPECT_EQ(RoundUpToMultipleOf<int32_t>(-7, 7), -7);
  EXPECT_EQ(RoundUpToMultipleOf<int32_t>(-8, 7), -7);
  EXPECT_EQ(RoundUpToMultipleOf<int32_t>(-14, 7), -14);
  EXPECT_EQ(RoundUpToMultipleOf<int32_t>(-15, 7), -14);
  EXPECT_EQ(RoundUpToMultipleOf<int32_t>(-14'006, 7), -14'000);

  // Test different int sizes
  EXPECT_EQ(RoundUpToMultipleOf<uint8_t>(13, 7), 14);
  EXPECT_EQ(RoundUpToMultipleOf<uint16_t>(13, 7), 14);
  EXPECT_EQ(RoundUpToMultipleOf<uint32_t>(13, 7), 14);
  EXPECT_EQ(RoundUpToMultipleOf<uint64_t>(13, 7), 14);

  EXPECT_EQ(RoundUpToMultipleOf<int8_t>(-13, 7), -7);
  EXPECT_EQ(RoundUpToMultipleOf<int16_t>(-13, 7), -7);
  EXPECT_EQ(RoundUpToMultipleOf<int32_t>(-13, 7), -7);
  EXPECT_EQ(RoundUpToMultipleOf<int64_t>(-13, 7), -7);
}

TEST(MathTests, TestBitsRequiredToRepresent)
{
  EXPECT_EQ(BitsRequiredToRepresent<int8_t>(-1), 8);
  EXPECT_EQ(BitsRequiredToRepresent<int32_t>(-1), 32);

  EXPECT_EQ(BitsRequiredToRepresent<uint8_t>(0), 0);
  EXPECT_EQ(BitsRequiredToRepresent<uint8_t>(1), 1);
  EXPECT_EQ(BitsRequiredToRepresent<uint8_t>(2), 2);
  EXPECT_EQ(BitsRequiredToRepresent<uint8_t>(3), 2);
  EXPECT_EQ(BitsRequiredToRepresent<uint8_t>(4), 3);
  EXPECT_EQ(BitsRequiredToRepresent<uint8_t>(7), 3);
  EXPECT_EQ(BitsRequiredToRepresent<uint8_t>(8), 4);

  EXPECT_EQ(BitsRequiredToRepresent(0xFFFF), 16);
  EXPECT_EQ(BitsRequiredToRepresent(0x10000), 17);
  EXPECT_EQ(BitsRequiredToRepresent(0xFFFF'FFFF), 32);
  EXPECT_EQ(BitsRequiredToRepresent(0xFFFF'FFFFu), 32);
  EXPECT_EQ(BitsRequiredToRepresent(0xFFFF'FFFF'FFFF'FFFFll), 64);
  EXPECT_EQ(BitsRequiredToRepresent(0xFFFF'FFFF'FFFF'FFFFull), 64);
}

TEST(MathTests, TestBitWidthOfEnum)
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

  EXPECT_EQ(BitWidthOfEnum(TestEnum1::Zero), 0);
  EXPECT_EQ(BitWidthOfEnum(TestEnum4::Three), 2);
  EXPECT_EQ(BitWidthOfEnum(TestEnum5::Four), 3);
  EXPECT_EQ(BitWidthOfEnum(TestEnum127::OneHundredAndTwentySeven), 7);
}

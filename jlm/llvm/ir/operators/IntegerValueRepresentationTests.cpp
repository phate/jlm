/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>

namespace jlm::llvm
{

TEST(IntegerValueRepresentationTest, WideArithmeticAndConversions)
{
  const IntegerValueRepresentation one(130, 1);
  const auto high = one.shl(100);
  EXPECT_EQ(high.shr(100), one);
  EXPECT_EQ(high.add(one).sub(one), high);
  EXPECT_EQ(
      high.mul(IntegerValueRepresentation(130, 3)).udiv(IntegerValueRepresentation(130, 3)),
      high);
  EXPECT_EQ(high.umod(IntegerValueRepresentation(130, 2)), IntegerValueRepresentation(130, 0));
  const IntegerValueRepresentation minusTwo(65, -2);
  EXPECT_EQ(minusTwo.sext(65), IntegerValueRepresentation(130, -2));
  EXPECT_EQ(minusTwo.zext(65).shr(65), IntegerValueRepresentation(130, 0));
  EXPECT_EQ(minusTwo.ashr(1), IntegerValueRepresentation(65, -1));
  EXPECT_EQ(minusTwo.sdiv(IntegerValueRepresentation(65, 2)), IntegerValueRepresentation(65, -1));
}

TEST(IntegerValueRepresentationTest, WordBoundariesAndBitwiseOperations)
{
  const IntegerValueRepresentation one(129, 1);
  const auto at63 = one.shl(63);
  const auto at64 = one.shl(64);
  const auto at128 = one.shl(128);
  EXPECT_EQ(at63.add(at64).land(at64), at64);
  EXPECT_EQ(at63.lor(at64).lxor(at64), at63);
  EXPECT_EQ(at128.shl(1), IntegerValueRepresentation(129, 0));
  EXPECT_EQ(at128.ashr(128), IntegerValueRepresentation(129, -1));
  EXPECT_EQ(at64.shr(64), one);
}

TEST(IntegerValueRepresentationTest, SignedDivisionAndRemainder)
{
  const IntegerValueRepresentation minusSeven(128, -7);
  const IntegerValueRepresentation three(128, 3);
  EXPECT_EQ(minusSeven.sdiv(three), IntegerValueRepresentation(128, -2));
  EXPECT_EQ(minusSeven.smod(three), IntegerValueRepresentation(128, -1));
  EXPECT_ANY_THROW(minusSeven.udiv(IntegerValueRepresentation(128, 0)));
  EXPECT_ANY_THROW(minusSeven.add(IntegerValueRepresentation(64, 1)));
  EXPECT_ANY_THROW(IntegerValueRepresentation(0, 0));
}

TEST(IntegerValueRepresentationTest, RejectsNonBinaryBits)
{
#ifdef JLM_ENABLE_ASSERTS
  EXPECT_DEATH(IntegerValueRepresentation("10X1"), "");
  EXPECT_DEATH(IntegerValueRepresentation("10D1"), "");
  EXPECT_DEATH(IntegerValueRepresentation::repeat(8, 'X'), "");
  EXPECT_DEATH(IntegerValueRepresentation::repeat(8, 'D'), "");
#endif
}

TEST(IntegerValueRepresentationTest, ExhaustiveSmallUnsignedValues)
{
  for (size_t width = 1; width <= 5; ++width)
  {
    const unsigned limit = 1u << width;
    for (unsigned a = 0; a < limit; ++a)
    {
      for (unsigned b = 0; b < limit; ++b)
      {
        const IntegerValueRepresentation x(width, a), y(width, b);
        EXPECT_EQ(x.add(y).to_uint(), (a + b) % limit);
        EXPECT_EQ(x.sub(y).to_uint(), (a + limit - b) % limit);
        EXPECT_EQ(x.mul(y).to_uint(), (a * b) % limit);
        EXPECT_EQ(x.ult(y), a < b ? '1' : '0');
        EXPECT_EQ(x.ule(y), a <= b ? '1' : '0');
        if (b != 0)
        {
          EXPECT_EQ(x.udiv(y).to_uint(), a / b);
          EXPECT_EQ(x.umod(y).to_uint(), a % b);
        }
      }
    }
  }
}

}

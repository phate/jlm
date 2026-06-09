/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_MATH_HPP
#define JLM_UTIL_MATH_HPP

#include <jlm/util/common.hpp>

#include <cstdint>
#include <type_traits>

namespace jlm::util
{

/**
 * Log2 for integers, rounding down when value is not a perfect power of two.
 * Any value less than 1 becomes -1.
 *
 * Examples:
 * Log2floor(4) = 2
 * Log2floor(7) = 2
 * Log2floor(8) = 3
 */
template<class T>
static constexpr int
log2Floor(T value)
{
  static_assert(std::is_integral_v<T>, "T must be integral type");
  if (value < 1)
    return -1;

  return 1 + log2Floor(value >> 1);
}

/**
 * Finds the smallest power of two that is at least as large as the given \p value.
 *
 * Examples:
 * RoundUpToPowerOf2(-10) == 1
 * RoundUpToPowerOf2(0) == 1
 * RoundUpToPowerOf2(1) == 1
 * RoundUpToPowerOf2(2) == 2
 * RoundUpToPowerOf2(7) == 8
 * RoundUpToPowerOf2(8) == 8
 * RoundUpToPowerOf2(9) == 16
 */
template<class T>
static constexpr T
RoundUpToPowerOf2(T value)
{
  // 2^0 == 1 is the lowest possible power of two
  if (value <= 1)
    return 1;

  return T(1) << (log2Floor(value - 1) + 1);
}

/**
 * Rounds the given \p value up to a multiple of \p multiple.
 * The multiple must be a strictly positive integer.
 *
 * Examples:
 * RoundUpToMultipleOf(3, 5) = 5
 * RoundUpToMultipleOf(5, 5) = 5
 * RoundUpToMultipleOf(6, 5) = 10
 * RoundUpToMultipleOf(0, 5) = 0
 * RoundUpToMultipleOf(-11, 5) = -10
 *
 * @return the result of rounding up, if value is not already a whole multiple
 */
template<class T>
static constexpr T
RoundUpToMultipleOf(T value, T multiple)
{
  const auto miss = value % multiple;
  if (miss < 0)
    return value - miss;
  if (miss == 0)
    return value;
  return value + multiple - miss;
}

/**
 * The number of bits needed to hold the given \p value.
 *
 * Examples:
 * BitsRequiredToRepresent(0)     = 0
 * BitsRequiredToRepresent(0b1)   = 1
 * BitsRequiredToRepresent(0b10)  = 2
 * BitsRequiredToRepresent(0b100) = 3
 * BitsRequiredToRepresent(0b111) = 3
 */
template<class T>
static constexpr int
BitsRequiredToRepresent(T value)
{
  using UnsignedT = std::make_unsigned_t<T>;
  return log2Floor(static_cast<UnsignedT>(value)) + 1;
}

/**
 * Calculates the number of bits needed to hold the integer representation of all enum values.
 * @param endValue the largest enum value, requires all other enum values to have a lower integer
 * value.
 */
template<class T>
static constexpr int
BitWidthOfEnum(T endValue)
{
  static_assert(std::is_enum_v<T>, "BitWidthOfEnum only takes enums");

  using UnderlyingT = std::underlying_type_t<T>;

  // To appease gcc warnings, the returned bit width is large enough to hold the endValue as well,
  // even if it is just a sentinel COUNT value
  return BitsRequiredToRepresent(static_cast<UnderlyingT>(endValue));
}

/**
 * Takes a 64-bit integer \p value and keeps the \p inputBit least significant bits.
 * All other bits are replaced by copies of the most significant kept input bit.
 *
 * Example:
 *   signExtendInteger(0b1010 0111, 8) = 0b1111 ...[48x 1s]... 1111 1010 0111
 *   signExtendInteger(0b1010 0111, 4) = 0b0000 ...[48x 0s]... 0000 0000 0111
 *
 * @param value the input value, which must fit in 64 bits
 * @param keepBits the number of least significant bits to keep as-is
 * @return the result of sign extending the truncated value back to a 64-bit value
 */
inline int64_t
truncateAndSignExtend(int64_t value, uint64_t keepBits)
{
  JLM_ASSERT(keepBits <= 64);
  const auto extendBits = 64 - keepBits;

  // Shift signed value left and right again to sign extend
  return (value << extendBits) >> extendBits;
}

/**
 * Takes a 64-bit integer \p value and keeps the \p inputBit least significant bits.
 * All other bits are set to 0.
 *
 * Example:
 *   zeroExtendInteger(0b1010 0111, 8) = 0b0000 ...[48x 0s]... 0000 1010 0111
 *   zeroExtendInteger(0b1010 0111, 4) = 0b0000 ...[48x 0s]... 0000 0000 0111
 *
 * @param value the input value, which must fit in 64 bits
 * @param keepBits the number of least significant bits to keep as-is
 * @return the result of sign extending the truncated value back to a 64-bit value
 */
inline int64_t
truncateAndZeroExtend(int64_t value, uint64_t keepBits)
{
  JLM_ASSERT(keepBits <= 64);
  const auto extendBits = 64 - keepBits;

  // Shift unsigned value left and right again to zero extend
  return (static_cast<uint64_t>(value) << extendBits) >> extendBits;
}

}

#endif // JLM_UTIL_MATH_HPP

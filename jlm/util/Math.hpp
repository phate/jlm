/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_MATH_HPP
#define JLM_UTIL_MATH_HPP

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
static inline constexpr int Log2Floor(T value) {
  if (value < 1)
    return -1;

  return 1 + Log2Floor(value >> 1);
}

/**
 * The number of bits needed to hold the given value
 *
 * Examples:
 * BitsRequiredToRepresent(0)     = 0
 * BitsRequiredToRepresent(0b1)   = 1
 * BitsRequiredToRepresent(0b10)  = 2
 * BitsRequiredToRepresent(0b100) = 3
 * BitsRequiredToRepresent(0b111) = 3
 */
template<class T>
static inline constexpr int BitsRequiredToRepresent(T value) {
  using UnsignedT = std::make_unsigned_t<T>;
  return Log2Floor(static_cast<UnsignedT>(value)) + 1;
}

/**
 * Calculates the number of bits needed to hold the integer representation of all enum values.
 * @param endValue the largest enum value, requires all other enum values to have a lower integer value.
 */
template<class T>
static inline constexpr int BitWidthOfEnum(T endValue) {
  static_assert(std::is_enum<T>::value, "BitWidthOfEnum only takes enums");

  using UnderlyingT = std::underlying_type_t<T>;

  // To appease gcc warnings, the returned bit width is large enough to hold the endValue as well,
  // even if it is just a sentinel COUNT value
  return BitsRequiredToRepresent(static_cast<UnderlyingT>(endValue));
}

}

#endif //JLM_UTIL_MATH_HPP

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
 * Log2 for integers, rounding down when value is not a perfect power of two
 *
 * Examples:
 * Log2floor(4) = 2
 * Log2floor(7) = 2
 * Log2floor(8) = 3
 */
template<class T>
static inline constexpr int Log2floor(T value) {
  if (value <= 1)
    return 0;

  return 1 + Log2floor(value >> 1);
}

/**
 * The number of bits needed to hold the given value
 *
 * Examples:
 * BitsRequiredToRepresent(0b1)   = 1
 * BitsRequiredToRepresent(0b10)  = 2
 * BitsRequiredToRepresent(0b100) = 3
 * BitsRequiredToRepresent(0b111) = 3
 */
template<class T>
static inline constexpr int BitsRequiredToRepresent(T value) {
  return Log2floor(value) + 1;
}

/**
 * Calculates the number of bits needed to hold the integer representation of all enum values.
 * Takes a sentinel end value as parameter, requires all other enum values to have a lower integer value.
 */
template<class T>
static inline constexpr int BitWidthOfEnum(T sentinelEndValue) {
  static_assert(std::is_enum<T>::value, "BitWidthOfEnum only takes enums");

  using UnderlyingT = std::underlying_type_t<T>;

  // To appease gcc warnings, the returned bit width is large enough to hold the sentinel value as well
  return BitsRequiredToRepresent(static_cast<UnderlyingT>(sentinelEndValue));
}

}

#endif //JLM_UTIL_MATH_HPP

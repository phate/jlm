

#ifndef JLM_UTIL_MATH_HPP
#define JLM_UTIL_MATH_HPP

#include <type_traits>

namespace jlm::util
{

/**
 * Log2 for integers, rounding up when value is not a perfect power of two
 */
template<class T>
static inline constexpr int Log2i(T value) {
  if (value <= 1)
    return 0;
  return 1 + Log2i(value >> 1);
}

/**
 * Calculates the number of bits needed to hold the integer representation of all enum values.
 * Takes a sentinel end value as parameter, requires all other enum values to have a lower integer value.
 */
template<class T>
static inline constexpr int BitWidthOfEnum(T sentinelEndValue) {
  static_assert(std::is_enum<T>::value, "BitWidthOfEnum only takes enums");

  using UnderlyingT = std::underlying_type_t<T>;
  return Log2i(static_cast<UnderlyingT>(sentinelEndValue));
}

}

#endif //JLM_UTIL_MATH_HPP

/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_HASH_HPP
#define JLM_UTIL_HASH_HPP

#include <functional>
#include <string_view>

namespace jlm::util
{

/**
 * Our own version of std::hash that also supports hashing std::pair
 */
template<typename T>
struct Hash : std::hash<T>
{
};

template<typename First, typename Second>
struct Hash<std::pair<First, Second>>
{
  std::size_t
  operator()(const std::pair<First, Second> & value) const noexcept
  {
    return Hash<First>()(value.first) ^ Hash<Second>()(value.second) << 1;
  }
};

/**
 * Combines multiple hash values given a seed value.
 *
 * @tparam Args The type of the hash values, i.e., std::size_t.
 * @param seed The seed value. It contains the combined hash values after the function invocation.
 * @param hash The first hash value.
 * @param args The other hash values.
 *
 * @see CombineHashes
 */
template<typename... Args>
void
CombineHashesWithSeed(std::size_t & seed, std::size_t hash, Args... args)
{
  seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  (CombineHashesWithSeed(seed, args), ...);
}

/**
 * Combines multiple hash values with the seed value 0.
 *
 * @tparam Args The type of the hash values, i.e, std::size_t.
 * @param hash The first hash value.
 * @param args The other hash values.
 * @return The combined hash values.
 *
 * @see CombineHashesWithSeed
 */
template<typename... Args>
std::size_t
CombineHashes(std::size_t hash, Args... args)
{
  std::size_t seed = 0;
  CombineHashesWithSeed(seed, hash, std::forward<Args>(args)...);
  return seed;
}

}

#endif

/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_BI_MAP_HPP
#define JLM_UTIL_BI_MAP_HPP

#include <jlm/util/common.hpp>

#include <unordered_map>

namespace jlm::util
{
/**
 * Represents a bijective mapping between keys and values, with fast loopups by both key and value.
 * Keys and values must be unique among themselves.
 * @tparam K the type of keys
 * @tparam V the type of values
 */
template<typename K, typename V>
class BiMap
{
  /**
   * Adds a key/value pair to only the reverse map.
   * @throws jlm::util::error if the value already exists in the BiMap.
   */
  void
  AddPairToReverse(K key, V value)
  {
    auto it = ReverseMap_.find(value);
    if (it != ReverseMap_.end())
      throw jlm::util::error("Value already exsists in BiMap");
    ReverseMap_.insert({ value, key });
  }

public:
  ~BiMap() noexcept = default;

  BiMap() = default;

  BiMap(const BiMap & other) = default;

  BiMap(BiMap && other) noexcept = default;

  BiMap &
  operator=(const BiMap & other) = default;

  BiMap &
  operator=(BiMap && other) = default;

  /**
   * Constructs a BiMap from an existing unordered_map.
   * @throws jlm::util::error if the values are not unique
   */
  explicit BiMap(const std::unordered_map<K, V> & forwardMap)
      : ForwardMap_(forwardMap),
        ReverseMap_()
  {
    for (auto & [key, value] : ForwardMap_)
      AddPairToReverse(key, value);
  }

  /**
   * Removes all key/value mappings from the BiMap.
   */
  void
  Clear() noexcept
  {
    ForwardMap_.clear();
    ReverseMap_.clear();
  }

  /**
   * Gets the number of key/value mappings in the BiMap.
   */
  [[nodiscard]] std::size_t
  Size() const noexcept
  {
    return ForwardMap_.size();
  }

  /**
   * Inserts a new key/value pair into the BiMap,
   * @param key the key to be inserted, with the associated value
   * @param value the value to be inserted, associated with the key
   * @throws jlm::util::error if the key or value already exists in the BiMap
   */
  void
  Insert(const K & key, const V & value)
  {
    auto it = ForwardMap_.find(key);
    if (it != ForwardMap_.end())
      throw jlm::util::error("Key already exsists in BiMap");
    AddPairToReverse(key, value);
    ForwardMap_.insert({ key, value });
  }

  /**
   * Checks for the existance of the given \p key in the BiMap
   * @param key the key to look for
   * @returns true if the key exists in the BiMap, otherwise false.
   */
  [[nodiscard]] bool
  HasKey(const K & key) const noexcept
  {
    return ForwardMap_.find(key) != ForwardMap_.end();
  }

  /**
   * Checks for the existance of the given \p value in the BiMap
   * @param value the value to look for
   * @returns true if the value exists in the BiMap, otherwise false.
   */
  [[nodiscard]] bool
  HasValue(const V & value) const noexcept
  {
    return ReverseMap_.find(value) != ReverseMap_.end();
  }

  /**
   * @param key the key to lookup in the BiMap
   * @returns the value associated with the given \p key
   * @throws jlm::util::error if the \p key does not exist in the BiMap
   */
  [[nodiscard]] const V &
  LookupKey(const K & key) const
  {
    auto it = ForwardMap_.find(key);
    if (it == ForwardMap_.end())
      throw jlm::util::error("Key not found in BiMap");
    return it->second;
  }

  /**
   * @param value the value to lookup in the BiMap
   * @returns the key associated with the given \p value
   * @throws jlm::util::error if the \p value does not exist in the BiMap
   */
  [[nodiscard]] const K &
  LookupValue(const V & value) const
  {
    auto it = ReverseMap_.find(value);
    if (it == ReverseMap_.end())
      throw jlm::util::error("Value not found in BiMap");
    return it->second;
  }

  /**
   * Removes a key/value pair from the BiMap, if the given \p key exists
   * @param key the key to be removed
   * @returns true if a key/value pair was removed
   */
  bool
  RemoveKey(const K & key)
  {
    auto it = ForwardMap_.find(key);
    if (it == ForwardMap_.end())
      return false;

    auto removed = ReverseMap_.erase(it->second);
    JLM_ASSERT(removed == 1);

    ForwardMap_.erase(it);
    return true;
  }

  /**
   * Removes a key/value pair from the BiMap, if the given \p value exists
   * @param value the value to be removed
   * @returns true if a key/value pair was removed
   */
  bool
  RemoveValue(const V & value)
  {
    auto it = ReverseMap_.find(value);
    if (it == ReverseMap_.end())
      return false;

    auto removed = ForwardMap_.erase(it->second);
    JLM_ASSERT(removed == 1);

    ReverseMap_.erase(it);
    return true;
  }

  /**
   * @returns the underlying mapping from keys to values
   */
  [[nodiscard]] const std::unordered_map<K, V> &
  GetForwardMap() const noexcept
  {
    return ForwardMap_;
  }

  /**
   * @returns the underlying mapping from values to keys
   */
  [[nodiscard]] const std::unordered_map<V, K> &
  GetReverseMap() const noexcept
  {
    return ReverseMap_;
  }

  /**
   * Shorthand for iterating over all key/value pairs in the BiMap
   * @returns a const iterator pointing to the first key/value pair in the ForwardMap
   */
  [[nodiscard]] typename std::unordered_map<K, V>::const_iterator
  begin() const noexcept
  {
    return ForwardMap_.cbegin();
  }

  /**
   * Shorthand for iterating over all key/value pairs
   * @returns a const iterator pointing to the end of the ForwardMap
   */
  [[nodiscard]] typename std::unordered_map<K, V>::const_iterator
  end() const noexcept
  {
    return ForwardMap_.cend();
  }

  /**
   * Compares the key/value mapping in two BiMaps.
   * @param other the BiMap to comapre against
   * @return true if the set of key/value pairs are identical across the two BiMaps
   */
  [[nodiscard]] bool
  operator==(const BiMap & other) const noexcept
  {
    // We only need to compare forward maps, as reverse maps are uniquely defined by the forward map
    return ForwardMap_ == other.ForwardMap_;
  }

  /**
   * Checks for inequality between two BiMaps.
   * \see operator==
   */
  [[nodiscard]] bool
  operator!=(const BiMap & other) const noexcept
  {
    return !operator==(other);
  }

private:
  std::unordered_map<K, V> ForwardMap_;
  std::unordered_map<V, K> ReverseMap_;
};

}

#endif // JLM_UTIL_BI_MAP_HPP

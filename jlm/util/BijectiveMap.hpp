/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_BIJECTIVE_MAP_HPP
#define JLM_UTIL_BIJECTIVE_MAP_HPP

#include <jlm/util/common.hpp>

#include <unordered_map>

namespace jlm::util
{
/**
 * Represents a bijective mapping between keys and values, with on average constant time lookups
 * by either key or value. Keys and values must be unique among themselves.
 * @tparam K the type of keys
 * @tparam V the type of values
 */
template<typename K, typename V>
class BijectiveMap
{
  using ForwardMapType = std::unordered_map<K, V>;
  using ReverseMapType = std::unordered_map<V, K>;

public:
  using ItemType = std::pair<const K, V>;

  class ConstIterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = ItemType;
    using difference_type = std::ptrdiff_t;
    using pointer = ItemType *;
    using reference = ItemType &;

    friend BijectiveMap;

    ConstIterator() = default;

    explicit ConstIterator(const typename ForwardMapType::const_iterator & it)
        : It_(it)
    {}

    const ItemType &
    operator*() const
    {
      return It_.operator*();
    }

    const ItemType *
    operator->() const
    {
      return It_.operator->();
    }

    ConstIterator &
    operator++()
    {
      ++It_;
      return *this;
    }

    ConstIterator
    operator++(int)
    {
      ConstIterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool
    operator==(const ConstIterator & other) const
    {
      return It_ == other.It_;
    }

    bool
    operator!=(const ConstIterator & other) const
    {
      return !operator==(other);
    }

  private:
    typename ForwardMapType::const_iterator It_;
  };

  ~BijectiveMap() noexcept = default;

  /**
   * Constructs an empty BijectiveMap
   */
  BijectiveMap() = default;

  BijectiveMap(const BijectiveMap & other) = default;

  BijectiveMap(BijectiveMap && other) noexcept = default;

  /**
   * Constructs a BijectiveMap from a range of std::pair<K, V>.
   * @throws jlm::util::error if any key or value is not unique
   */
  template<class InputIt>
  explicit BijectiveMap(InputIt first, InputIt last)
      : ForwardMap_(),
        ReverseMap_()
  {
    for (auto it = first; it != last; ++it)
      InsertOrThrow(it->first, it->second);
  }

  /**
   * Constructs a BijectiveMap from an initializer list of std::pair<K, V>
   * \param init the initializer list
   * @throws jlm::util::error if any key or value is not unique
   */
  BijectiveMap(std::initializer_list<ItemType> init)
      : BijectiveMap(init.begin(), init.end())
  {}

  BijectiveMap &
  operator=(const BijectiveMap & other) = default;

  BijectiveMap &
  operator=(BijectiveMap && other) = default;

  /**
   * Removes all key/value mappings from the BijectiveMap.
   */
  void
  Clear() noexcept
  {
    ForwardMap_.clear();
    ReverseMap_.clear();
  }

  /**
   * Gets the number of key/value mappings in the BijectiveMap.
   */
  [[nodiscard]] std::size_t
  Size() const noexcept
  {
    JLM_ASSERT(ForwardMap_.size() == ReverseMap_.size());
    return ForwardMap_.size();
  }

  /**
   * Inserts a new key/value pair into the BijectiveMap,
   * if neither the key nor the value are present in the map already.
   * @param key the key to be inserted, with the associated value
   * @param value the value to be inserted, associated with the key
   * @return true if the key/value pair is inserted, otherwise false.
   */
  bool
  Insert(const K & key, const V & value)
  {
    if (HasKey(key) || HasValue(value))
      return false;

    ForwardMap_.insert({ key, value });
    ReverseMap_.insert({ value, key });
    return true;
  }

  /**
   * Tries to insert all key/value mappings provided by an iterator range
   * @tparam IteratorType the iterator type, a forward iterator with std::pair<K, V> value type.
   * \param begin an iterator pointing to the first key/value pair to be inserted
   * \param end an iterator pointing just past the last key/value pair to be inserted
   * \return the number of key/value pairs inserted
   */
  template<typename IteratorType>
  size_t
  InsertPairs(IteratorType begin, IteratorType end)
  {
    static_assert(
        std::is_base_of_v<std::forward_iterator_tag, typename IteratorType::iterator_category>);

    size_t inserted = 0;
    while (begin != end)
    {
      if (Insert(begin->first, begin->second))
        inserted++;
      ++begin;
    }

    return inserted;
  }

  /**
   * Inserts a new key/value pair into the BijectiveMap.
   * Fails if either the key or value are already present in the map.
   * @param key the key to be inserted, with the associated value
   * @param value the value to be inserted, associated with the key
   * @throws jlm::util::error if the key or value already exists in the map
   */
  void
  InsertOrThrow(const K & key, const V & value)
  {
    if (!Insert(key, value))
      throw Error("Key or value were already present in the BijectiveMap");
  }

  /**
   * Checks for the existance of the given \p key in the BijectiveMap
   * @param key the key to look for
   * @return true if the key exists in the BijectiveMap, otherwise false.
   */
  [[nodiscard]] bool
  HasKey(const K & key) const noexcept
  {
    return ForwardMap_.count(key);
  }

  /**
   * Checks for the existance of the given \p value in the BijectiveMap
   * @param value the value to look forsome
   * @return true if the value exists in the BijectiveMap, otherwise false.
   */
  [[nodiscard]] bool
  HasValue(const V & value) const noexcept
  {
    return ReverseMap_.count(value);
  }

  /**
   * @param key the key to lookup in the BijectiveMap
   * @return the value associated with the given \p key
   * @throws jlm::util::error if the \p key does not exist in the BijectiveMap
   */
  [[nodiscard]] const V &
  LookupKey(const K & key) const
  {
    auto it = ForwardMap_.find(key);
    if (it == ForwardMap_.end())
      throw Error("Key not found in BijectiveMap");
    return it->second;
  }

  /**
   * @param value the value to lookup in the BijectiveMap
   * @return the key associated with the given \p value
   * @throws jlm::util::error if the \p value does not exist in the BijectiveMap
   */
  [[nodiscard]] const K &
  LookupValue(const V & value) const
  {
    auto it = ReverseMap_.find(value);
    if (it == ReverseMap_.end())
      throw Error("Value not found in BijectiveMap");
    return it->second;
  }

  /**
   * Provides an iterator pointing to the first key/value mappings in the BijectiveMap,
   * that can be used to iterate over all pairs, with no ordering guarantees.
   * @return ConstIterator pointing to the first key/value mapping
   */
  [[nodiscard]] ConstIterator
  begin() const noexcept
  {
    return ConstIterator(ForwardMap_.cbegin());
  }

  /**
   * Provides an iterator pointing to the element following the last key/value mapping.
   * It is a placeholder, and must not be accessed.
   * @return ConstIterator pointing past the last key/value mapping
   */
  [[nodiscard]] ConstIterator
  end() const noexcept
  {
    return ConstIterator(ForwardMap_.cend());
  }

  /**
   * Given an iterator pointing to a key/value pair in the BijectiveMap, removes that pair.
   * The iterator must be valid, and not point to end().
   * Only the removed iterator is affected by this operation, all other iterators stay valid.
   * @param it an iterator pointing to the element to be removed
   * @return ConstIterator pointing to the key/value pair immediately following the removed pair
   */
  ConstIterator
  Erase(ConstIterator it)
  {
    const size_t removed = ReverseMap_.erase(it->second);
    JLM_ASSERT(removed == 1);
    const auto nextForwardIt = ForwardMap_.erase(it.It_);
    return ConstIterator(nextForwardIt);
  }

  /**
   * Removes a key/value pair from the BijectiveMap, if the given \p key exists
   * @param key the key to be removed
   * @return true if a key/value pair was removed
   */
  bool
  RemoveKey(const K & key)
  {
    auto it = ForwardMap_.find(key);
    if (it == ForwardMap_.end())
      return false;

    Erase(ConstIterator(it));
    return true;
  }

  /**
   * Removes a key/value pair from the BijectiveMap, if the given \p value exists
   * @param value the value to be removed
   * @return true if a key/value pair was removed
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
   * Removes all key/value pairs in the BijectiveMap that satisfy the predicate \p match.
   * \tparam F a functor with the signature (const K &, const V &) -> bool
   * \param match an instance of F, to be invoked on each pair
   * @return the number of key/value pairs that were removed
   */
  template<typename F>
  size_t
  RemoveWhere(const F & match)
  {
    size_t removed = 0;
    auto it = begin();
    while (it != end())
    {
      if (match(it->first, it->second))
      {
        it = Erase(it);
        removed++;
      }
      else
      {
        ++it;
      }
    }
    return removed;
  }

  /**
   * Removes all key/value pairs where the key satisfies the predicate \p match.
   * \tparam F a functor with the signature (const K &) -> bool
   * \param match an instance of F, to be invoked on each key
   * @return the number of key/value pairs that were removed
   */
  template<typename F>
  size_t
  RemoveKeysWhere(const F & match)
  {
    return RemoveWhere(
        [&](const K & key, const V &)
        {
          return match(key);
        });
  }

  /**
   * Removes all key/value pairs where the value satisfies the predicate \p match.
   * \tparam F a functor with the signature (const V &) -> bool
   * \param match an instance of F, to be invoked on each value
   * @return the number of key/value pairs that were removed
   */
  template<typename F>
  size_t
  RemoveValuesWhere(const F & match)
  {
    return RemoveWhere(
        [&](const K &, const V & value)
        {
          return match(value);
        });
  }

  /**
   * Compares the key/value mapping in two BijectiveMaps.
   * @param other the BijectiveMap to comapre against
   * @return true if the set of key/value pairs are identical across the two BijectiveMaps
   */
  [[nodiscard]] bool
  operator==(const BijectiveMap & other) const noexcept
  {
    // We only need to compare forward maps, as reverse maps are uniquely defined by the forward map
    return ForwardMap_ == other.ForwardMap_;
  }

  /**
   * Checks for inequality between two BijectiveMaps.
   * \see operator==
   */
  [[nodiscard]] bool
  operator!=(const BijectiveMap & other) const noexcept
  {
    return !operator==(other);
  }

private:
  ForwardMapType ForwardMap_;
  ReverseMapType ReverseMap_;
};

}

#endif // JLM_UTIL_BIJECTIVE_MAP_HPP

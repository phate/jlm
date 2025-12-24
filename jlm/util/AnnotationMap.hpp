/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_ANNOTATION_MAP_HPP
#define JLM_UTIL_ANNOTATION_MAP_HPP

#include <jlm/util/common.hpp>
#include <jlm/util/iterator_range.hpp>

#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace jlm::util
{

/**
 * Represents a simple key-value pair with a label and a value of type std::string, int64_t,
 * uint64_t, or double.
 */
class Annotation final
{
  using AnnotationValue = std::variant<std::string, int64_t, uint64_t, double>;

public:
  Annotation(std::string_view label, std::string value)
      : Label_(std::move(label)),
        Value_(std::move(value))
  {}

  Annotation(std::string_view label, int64_t value)
      : Label_(std::move(label)),
        Value_(std::move(value))
  {}

  Annotation(std::string_view label, uint64_t value)
      : Label_(std::move(label)),
        Value_(std::move(value))
  {}

  Annotation(std::string_view label, double value)
      : Label_(std::move(label)),
        Value_(std::move(value))
  {}

  /**
   * Gets the label of the annotation.
   */
  [[nodiscard]] const std::string_view &
  Label() const noexcept
  {
    return Label_;
  }

  /**
   * Gets the value of the annotation. Requires the annotation value to be of type \p TValue.
   */
  template<typename TValue>
  [[nodiscard]] const TValue &
  Value() const
  {
    return std::get<TValue>(Value_);
  }

  /**
   * Checks if the type of the annotation value is equivalent to \p TValue.
   *
   * @return True if the value type if equivalent to \p TValue, otherwise false.
   */
  template<typename TValue>
  [[nodiscard]] bool
  HasValueType() const noexcept
  {
    return std::holds_alternative<TValue>(Value_);
  }

  bool
  operator==(const Annotation & other) const noexcept
  {
    return Label_ == other.Label_ && Value_ == other.Value_;
  }

  bool
  operator!=(const Annotation & other) const noexcept
  {
    return !(*this == other);
  }

private:
  std::string_view Label_ = {};
  AnnotationValue Value_ = {};
};

/**
 * Represents a simple map that associates pointers with Annotation%s.
 */
class AnnotationMap final
{
  using AnnotationMapType = std::unordered_map<const void *, std::vector<Annotation>>;

  class ConstIterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Annotation;
    using difference_type = std::ptrdiff_t;
    using pointer = Annotation *;
    using reference = Annotation &;

  private:
    friend AnnotationMap;

    explicit ConstIterator(const typename AnnotationMapType::const_iterator & it)
        : It_(it)
    {}

  public:
    [[nodiscard]] const std::vector<Annotation> &
    Annotations() const noexcept
    {
      return It_.operator->()->second;
    }

    const std::vector<Annotation> &
    operator*() const
    {
      return Annotations();
    }

    const std::vector<Annotation> *
    operator->() const
    {
      return &Annotations();
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
    typename AnnotationMapType::const_iterator It_ = {};
  };

  using AnnotationRange = IteratorRange<AnnotationMap::ConstIterator>;

public:
  /**
   * Retrieves all annotations.
   *
   * @return An iterator_range of all the annotations.
   */
  [[nodiscard]] AnnotationRange
  Annotations() const
  {
    return { ConstIterator(Map_.begin()), ConstIterator(Map_.end()) };
  }

  /**
   * Checks if an annotation with the given \p key exists.
   *
   * @return True if the annotation exists, otherwise false.
   */
  [[nodiscard]] bool
  HasAnnotations(const void * key) const noexcept
  {
    return Map_.find(key) != Map_.end();
  }

  /**
   * Retrieves the annotation for the given \p key. The key must exist.
   *
   * @return A reference to an instance of Annotation.
   */
  [[nodiscard]] const std::vector<Annotation> &
  GetAnnotations(const void * key) const noexcept
  {
    JLM_ASSERT(HasAnnotations(key));
    return Map_.at(key);
  }

  /**
   * Adds \p annotation with the given \p key to the map.
   */
  void
  AddAnnotation(const void * key, Annotation annotation)
  {
    Map_[key].emplace_back(std::move(annotation));
  }

private:
  AnnotationMapType Map_ = {};
};

}

#endif

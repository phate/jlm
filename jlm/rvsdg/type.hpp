/*
 * Copyright 2010 2011 2012 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_TYPE_HPP
#define JLM_RVSDG_TYPE_HPP

#include <memory>
#include <string>

namespace jlm::rvsdg
{

/**
 * \brief The kinds of types supported in rvsdg
 *
 * Note: there might be reasons to make this extensible eventually.
 */
enum class TypeKind
{
  /**
   * \brief Designate a value type
   *
   * Value types represent things that can be created, copied and
   * destroyed at will.
   */
  Value,
  /**
   * \brief Designate a state type
   *
   * State types represent things that exist in mutable form.
   * They cannot be copied, but objects of this kind must
   * be used linearly.
   */
  State
};

class Type
{
public:
  virtual ~Type() noexcept;

protected:
  inline constexpr Type() noexcept
  {}

public:
  virtual bool
  operator==(const jlm::rvsdg::Type & other) const noexcept = 0;

  inline bool
  operator!=(const jlm::rvsdg::Type & other) const noexcept
  {
    return !(*this == other);
  }

  [[nodiscard]] virtual std::string
  debug_string() const = 0;

  /**
   * Computes a hash value for the instance of the type.
   *
   * @return A hash value.
   */
  [[nodiscard]] virtual std::size_t
  ComputeHash() const noexcept = 0;

  /**
   * \brief Return the kind of this type
   */
  virtual TypeKind
  Kind() const noexcept = 0;
};

template<class T>
static inline bool
is(const jlm::rvsdg::Type & type) noexcept
{
  static_assert(
      std::is_base_of<jlm::rvsdg::Type, T>::value,
      "Template parameter T must be derived from jlm::rvsdg::Type.");

  return dynamic_cast<const T *>(&type) != nullptr;
}

template<class T>
static inline bool
is(const std::shared_ptr<const jlm::rvsdg::Type> & type) noexcept
{
  static_assert(
      std::is_base_of<jlm::rvsdg::Type, T>::value,
      "Template parameter T must be derived from jlm::rvsdg::Type.");

  return dynamic_cast<const T *>(type.get()) != nullptr;
}

}

#endif

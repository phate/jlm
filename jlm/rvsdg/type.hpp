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

class type
{
public:
  virtual ~type() noexcept;

protected:
  inline constexpr type() noexcept
  {}

public:
  virtual bool
  operator==(const jlm::rvsdg::type & other) const noexcept = 0;

  inline bool
  operator!=(const jlm::rvsdg::type & other) const noexcept
  {
    return !(*this == other);
  }

  virtual std::string
  debug_string() const = 0;

  // FIXME: add documentation
  [[nodiscard]] virtual std::size_t
  ComputeHash() const noexcept = 0;
};

class valuetype : public jlm::rvsdg::type
{
public:
  virtual ~valuetype() noexcept;

protected:
  inline constexpr valuetype() noexcept
      : jlm::rvsdg::type()
  {}
};

class statetype : public jlm::rvsdg::type
{
public:
  virtual ~statetype() noexcept;

protected:
  inline constexpr statetype() noexcept
      : jlm::rvsdg::type()
  {}
};

template<class T>
static inline bool
is(const jlm::rvsdg::type & type) noexcept
{
  static_assert(
      std::is_base_of<jlm::rvsdg::type, T>::value,
      "Template parameter T must be derived from jlm::rvsdg::type.");

  return dynamic_cast<const T *>(&type) != nullptr;
}

template<class T>
static inline bool
is(const std::shared_ptr<const jlm::rvsdg::type> & type) noexcept
{
  static_assert(
      std::is_base_of<jlm::rvsdg::type, T>::value,
      "Template parameter T must be derived from jlm::rvsdg::type.");

  return dynamic_cast<const T *>(type.get()) != nullptr;
}

}

#endif

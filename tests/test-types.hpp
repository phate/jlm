/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef TEST_TEST_TYPES_HPP
#define TEST_TEST_TYPES_HPP

#include <jlm/rvsdg/type.hpp>

namespace jlm::tests
{

class valuetype final : public rvsdg::valuetype
{
public:
  virtual ~valuetype();

  inline constexpr valuetype() noexcept
      : rvsdg::valuetype()
  {}

  virtual std::string
  debug_string() const override;

  virtual bool
  operator==(const rvsdg::type & other) const noexcept override;

  virtual std::shared_ptr<const jlm::rvsdg::type>
  copy() const override;
};

class statetype final : public rvsdg::statetype
{
public:
  virtual ~statetype();

  inline constexpr statetype() noexcept
      : rvsdg::statetype()
  {}

  virtual std::string
  debug_string() const override;

  virtual bool
  operator==(const rvsdg::type & other) const noexcept override;

  virtual std::shared_ptr<const jlm::rvsdg::type>
  copy() const override;
};

}

#endif

/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_OPERATION_HPP
#define JLM_RVSDG_OPERATION_HPP

#include <jlm/rvsdg/type.hpp>

#include <memory>
#include <string>
#include <vector>

namespace jlm::rvsdg
{

class graph;
class node;
class node_normal_form;
class output;
class region;
class simple_normal_form;
class structural_normal_form;

/* port */

class port
{
public:
  virtual ~port();

  port(const jlm::rvsdg::type & type);

  port(std::unique_ptr<jlm::rvsdg::type> type);

  inline port(const port & other)
      : type_(other.type_->copy())
  {}

  inline port(port && other)
      : type_(std::move(other.type_))
  {}

  inline port &
  operator=(const port & other)
  {
    if (&other == this)
      return *this;

    type_ = other.type_->copy();

    return *this;
  }

  inline port &
  operator=(port && other)
  {
    if (&other == this)
      return *this;

    type_ = std::move(other.type_);

    return *this;
  }

  virtual bool
  operator==(const port &) const noexcept;

  inline bool
  operator!=(const port & other) const noexcept
  {
    return !(*this == other);
  }

  inline const jlm::rvsdg::type &
  type() const noexcept
  {
    return *type_;
  }

  virtual std::unique_ptr<port>
  copy() const;

private:
  std::unique_ptr<jlm::rvsdg::type> type_;
};

/* operation */

class operation
{
public:
  virtual ~operation() noexcept;

  virtual bool
  operator==(const operation & other) const noexcept = 0;

  virtual std::string
  debug_string() const = 0;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const = 0;

  inline bool
  operator!=(const operation & other) const noexcept
  {
    return !(*this == other);
  }

  static jlm::rvsdg::node_normal_form *
  normal_form(jlm::rvsdg::graph * graph) noexcept;
};

template<class T>
static inline bool
is(const jlm::rvsdg::operation & operation) noexcept
{
  static_assert(
      std::is_base_of<jlm::rvsdg::operation, T>::value,
      "Template parameter T must be derived from jlm::rvsdg::operation.");

  return dynamic_cast<const T *>(&operation) != nullptr;
}

/* simple operation */

class simple_op : public operation
{
public:
  virtual ~simple_op();

  inline simple_op(
      const std::vector<jlm::rvsdg::port> & operands,
      const std::vector<jlm::rvsdg::port> & results)
      : results_(results),
        operands_(operands)
  {}

  size_t
  narguments() const noexcept;

  const jlm::rvsdg::port &
  argument(size_t index) const noexcept;

  size_t
  nresults() const noexcept;

  const jlm::rvsdg::port &
  result(size_t index) const noexcept;

  static jlm::rvsdg::simple_normal_form *
  normal_form(jlm::rvsdg::graph * graph) noexcept;

private:
  std::vector<jlm::rvsdg::port> results_;
  std::vector<jlm::rvsdg::port> operands_;
};

/* structural operation */

class structural_op : public operation
{
public:
  virtual bool
  operator==(const operation & other) const noexcept override;

  static jlm::rvsdg::structural_normal_form *
  normal_form(jlm::rvsdg::graph * graph) noexcept;
};

}

#endif

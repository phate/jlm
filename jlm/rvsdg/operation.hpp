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
class Region;
class simple_normal_form;
class structural_normal_form;

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

  simple_op(
      std::vector<std::shared_ptr<const jlm::rvsdg::Type>> operands,
      std::vector<std::shared_ptr<const jlm::rvsdg::Type>> results)
      : operands_(std::move(operands)),
        results_(std::move(results))
  {}

  size_t
  narguments() const noexcept;

  [[nodiscard]] const std::shared_ptr<const rvsdg::Type> &
  argument(size_t index) const noexcept;

  size_t
  nresults() const noexcept;

  [[nodiscard]] const std::shared_ptr<const rvsdg::Type> &
  result(size_t index) const noexcept;

  static jlm::rvsdg::simple_normal_form *
  normal_form(jlm::rvsdg::graph * graph) noexcept;

private:
  std::vector<std::shared_ptr<const rvsdg::Type>> operands_;
  std::vector<std::shared_ptr<const rvsdg::Type>> results_;
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

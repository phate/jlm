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

class Graph;
class Node;
class Output;
class Region;

class Operation
{
public:
  virtual ~Operation() noexcept;

  virtual bool
  operator==(const Operation & other) const noexcept = 0;

  virtual std::string
  debug_string() const = 0;

  [[nodiscard]] virtual std::unique_ptr<Operation>
  copy() const = 0;

  inline bool
  operator!=(const Operation & other) const noexcept
  {
    return !(*this == other);
  }
};

template<class T>
static inline bool
is(const Operation & operation) noexcept
{
  static_assert(
      std::is_base_of<Operation, T>::value,
      "Template parameter T must be derived from jlm::rvsdg::operation.");

  return dynamic_cast<const T *>(&operation) != nullptr;
}

class SimpleOperation : public Operation
{
public:
  ~SimpleOperation() noexcept override;

  SimpleOperation(
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

private:
  std::vector<std::shared_ptr<const rvsdg::Type>> operands_;
  std::vector<std::shared_ptr<const rvsdg::Type>> results_;
};

class StructuralOperation : public Operation
{
public:
  virtual bool
  operator==(const Operation & other) const noexcept override;
};

}

#endif

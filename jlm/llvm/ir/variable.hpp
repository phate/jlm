/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_VARIABLE_HPP
#define JLM_LLVM_IR_VARIABLE_HPP

#include <jlm/llvm/ir/linkage.hpp>
#include <jlm/rvsdg/type.hpp>
#include <jlm/util/strfmt.hpp>

#include <memory>
#include <sstream>

namespace jlm::llvm
{

/* variable */

class variable
{
public:
  virtual ~variable() noexcept;

  variable(std::shared_ptr<const jlm::rvsdg::Type> type, const std::string & name)
      : name_(name),
        type_(std::move(type))
  {}

  variable(variable && other)
      : name_(std::move(other.name_)),
        type_(std::move(other.type_))
  {}

  variable &
  operator=(variable && other)
  {
    if (this == &other)
      return *this;

    name_ = std::move(other.name_);
    type_ = std::move(other.type_);

    return *this;
  }

  virtual std::string
  debug_string() const;

  inline const std::string &
  name() const noexcept
  {
    return name_;
  }

  inline const jlm::rvsdg::Type &
  type() const noexcept
  {
    return *type_;
  }

  inline const std::shared_ptr<const jlm::rvsdg::Type>
  Type() const noexcept
  {
    return type_;
  }

private:
  std::string name_;
  std::shared_ptr<const jlm::rvsdg::Type> type_;
};

template<class T>
static inline bool
is(const llvm::variable * variable) noexcept
{
  static_assert(
      std::is_base_of<llvm::variable, T>::value,
      "Template parameter T must be derived from jlm::variable.");

  return dynamic_cast<const T *>(variable) != nullptr;
}

/* top level variable */

class gblvariable : public variable
{
public:
  virtual ~gblvariable();

  inline gblvariable(std::shared_ptr<const jlm::rvsdg::Type> type, const std::string & name)
      : variable(std::move(type), name)
  {}
};

}

#endif

/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_NULLARY_HPP
#define JLM_RVSDG_NULLARY_HPP

#include <jlm/rvsdg/node-normal-form.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/util/common.hpp>

namespace jlm::rvsdg
{

class output;

/**
  \brief Nullary operator (operator taking no formal arguments)
*/
class nullary_op : public simple_op
{
public:
  virtual ~nullary_op() noexcept;

  inline explicit nullary_op(std::shared_ptr<const jlm::rvsdg::Type> result)
      : simple_op({}, { std::move(result) })
  {}
};

template<typename Type, typename ValueRepr>
struct default_type_of_value
{
  Type
  operator()(const ValueRepr &) const noexcept
  {
    return Type();
  }
};

/* Template to represent a domain-specific constant. Instances are fully
 * characterised by the value they contain.
 *
 * Template argument requirements:
 * - Type: type class of the constants represented
 * - ValueRepr: representation of values
 * - FormatValue: functional that takes a ValueRepr instance and returns
 *   as std::string a human-readable representation of the value
 * - TypeOfValue: functional that takes a ValueRepr instance and returns
 *   the Type instances corresponding to this value (in case the type
 *   class is polymorphic) */
template<typename Type, typename ValueRepr, typename FormatValue, typename TypeOfValue>
class domain_const_op final : public nullary_op
{
public:
  typedef ValueRepr value_repr;

  virtual ~domain_const_op() noexcept
  {}

  inline domain_const_op(const value_repr & value)
      : nullary_op(TypeOfValue()(value)),
        value_(value)
  {}

  inline domain_const_op(const domain_const_op & other) = default;

  inline domain_const_op(domain_const_op && other) = default;

  virtual bool
  operator==(const operation & other) const noexcept override
  {
    auto op = dynamic_cast<const domain_const_op *>(&other);
    return op && op->value_ == value_;
  }

  virtual std::string
  debug_string() const override
  {
    return FormatValue()(value_);
  }

  inline const value_repr &
  value() const noexcept
  {
    return value_;
  }

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new domain_const_op(*this));
  }

  static inline jlm::rvsdg::output *
  create(rvsdg::Region * region, const value_repr & vr)
  {
    domain_const_op op(vr);
    return simple_node::create_normalized(region, op, {})[0];
  }

private:
  value_repr value_;
};

}

#endif

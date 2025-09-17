/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_NULLARY_HPP
#define JLM_RVSDG_NULLARY_HPP

#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::rvsdg
{

class Output;

/**
  \brief Nullary operator (operator taking no formal arguments)
*/
class NullaryOperation : public SimpleOperation
{
public:
  ~NullaryOperation() noexcept override;

  explicit NullaryOperation(std::shared_ptr<const Type> resultType)
      : SimpleOperation({}, { std::move(resultType) })
  {}
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
class DomainConstOperation final : public NullaryOperation
{
public:
  typedef ValueRepr value_repr;

  ~DomainConstOperation() noexcept override = default;

  explicit DomainConstOperation(const value_repr & value)
      : NullaryOperation(TypeOfValue()(value)),
        value_(value)
  {}

  DomainConstOperation(const DomainConstOperation & other) = default;

  DomainConstOperation(DomainConstOperation && other) = default;

  bool
  operator==(const Operation & other) const noexcept override
  {
    auto op = dynamic_cast<const DomainConstOperation *>(&other);
    return op && op->value_ == value_;
  }

  [[nodiscard]] std::string
  debug_string() const override
  {
    return FormatValue()(value_);
  }

  inline const value_repr &
  value() const noexcept
  {
    return value_;
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<DomainConstOperation>(*this);
  }

  static inline jlm::rvsdg::Output *
  create(rvsdg::Region * region, const value_repr & vr)
  {
    return CreateOpNode<DomainConstOperation>(*region, vr).output(0);
  }

private:
  value_repr value_;
};

}

#endif

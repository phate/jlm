/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/util/Hash.hpp>

namespace jlm::rvsdg
{

/* control constant */

// explicit instantiation
template class domain_const_op<ControlType, ctlvalue_repr, ctlformat_value, ctltype_of_value>;

ControlType::~ControlType() noexcept = default;

ControlType::ControlType(size_t nalternatives)
    : StateType(),
      nalternatives_(nalternatives)
{}

std::string
ControlType::debug_string() const
{
  return jlm::util::strfmt("ctl(", nalternatives_, ")");
}

bool
ControlType::operator==(const Type & other) const noexcept
{
  auto type = dynamic_cast<const ControlType *>(&other);
  return type && type->nalternatives_ == nalternatives_;
}

std::size_t
ControlType::ComputeHash() const noexcept
{
  auto typeHash = typeid(ControlType).hash_code();
  auto numAlternativesHash = std::hash<size_t>()(nalternatives_);
  return util::CombineHashes(typeHash, numAlternativesHash);
}

std::shared_ptr<const ControlType>
ControlType::Create(std::size_t nalternatives)
{
  static const ControlType static_instances[4] = {
    // ControlType(0) is not valid, but put it in here so
    // the static array indexing works correctly
    ControlType(0),
    ControlType(1),
    ControlType(2),
    ControlType(3)
  };

  if (nalternatives < 4)
  {
    if (nalternatives == 0)
    {
      throw jlm::util::error("Alternatives of a control type must be non-zero.");
    }
    return std::shared_ptr<const ControlType>(
        std::shared_ptr<void>(),
        &static_instances[nalternatives]);
  }
  else
  {
    return std::make_shared<ControlType>(nalternatives);
  }
}

/* control value representation */

ctlvalue_repr::ctlvalue_repr(size_t alternative, size_t nalternatives)
    : alternative_(alternative),
      nalternatives_(nalternatives)
{
  if (alternative >= nalternatives)
    throw jlm::util::error("Alternative is bigger than the number of possible alternatives.");
}

/* match operator */

match_op::~match_op() noexcept
{}

match_op::match_op(
    size_t nbits,
    const std::unordered_map<uint64_t, uint64_t> & mapping,
    uint64_t default_alternative,
    size_t nalternatives)
    : UnaryOperation(bittype::Create(nbits), ControlType::Create(nalternatives)),
      default_alternative_(default_alternative),
      mapping_(mapping)
{}

bool
match_op::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const match_op *>(&other);
  return op && op->default_alternative_ == default_alternative_ && op->mapping_ == mapping_
      && op->nbits() == nbits() && op->nalternatives() == nalternatives();
}

unop_reduction_path_t
match_op::can_reduce_operand(const jlm::rvsdg::Output * arg) const noexcept
{
  if (is<bitconstant_op>(producer(arg)))
    return unop_reduction_constant;

  return unop_reduction_none;
}

jlm::rvsdg::Output *
match_op::reduce_operand(unop_reduction_path_t path, jlm::rvsdg::Output * arg) const
{
  if (path == unop_reduction_constant)
  {
    auto op = static_cast<const bitconstant_op &>(producer(arg)->GetOperation());
    return jlm::rvsdg::control_constant(
        arg->region(),
        nalternatives(),
        alternative(op.value().to_uint()));
  }

  return nullptr;
}

std::string
match_op::debug_string() const
{
  std::string str("[");
  for (const auto & pair : mapping_)
    str += jlm::util::strfmt(pair.first, " -> ", pair.second, ", ");
  str += jlm::util::strfmt(default_alternative_, "]");

  return "MATCH" + str;
}

std::unique_ptr<Operation>
match_op::copy() const
{
  return std::make_unique<match_op>(*this);
}

jlm::rvsdg::Output *
match(
    size_t nbits,
    const std::unordered_map<uint64_t, uint64_t> & mapping,
    uint64_t default_alternative,
    size_t nalternatives,
    jlm::rvsdg::Output * operand)
{
  return CreateOpNode<match_op>({ operand }, nbits, mapping, default_alternative, nalternatives)
      .output(0);
}

jlm::rvsdg::Output *
control_constant(rvsdg::Region * region, size_t nalternatives, size_t alternative)
{
  return CreateOpNode<ctlconstant_op>(*region, ctlvalue_repr(alternative, nalternatives)).output(0);
}

}

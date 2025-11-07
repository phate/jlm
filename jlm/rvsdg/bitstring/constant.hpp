/*
 * Copyright 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_BITSTRING_CONSTANT_HPP
#define JLM_RVSDG_BITSTRING_CONSTANT_HPP

#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/bitstring/value-representation.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/nullary.hpp>
#include <jlm/rvsdg/simple-node.hpp>

#include <vector>

namespace jlm::rvsdg
{

class BitConstantOperation final : public NullaryOperation
{
public:
  ~BitConstantOperation() noexcept override;

  explicit BitConstantOperation(BitValueRepresentation value);

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] const BitValueRepresentation &
  value() const noexcept
  {
    return value_;
  }

  static Output *
  create(Region * region, BitValueRepresentation value)
  {
    return CreateOpNode<BitConstantOperation>(*region, std::move(value)).output(0);
  }

private:
  BitValueRepresentation value_;
};

static inline jlm::rvsdg::Output *
create_bitconstant(rvsdg::Region * region, size_t nbits, int64_t value)
{
  return BitConstantOperation::create(region, { nbits, value });
}

static inline jlm::rvsdg::Output *
create_bitconstant_undefined(rvsdg::Region * region, size_t nbits)
{
  std::string s(nbits, 'X');
  return BitConstantOperation::create(region, BitValueRepresentation(s.c_str()));
}

static inline jlm::rvsdg::Output *
create_bitconstant_defined(rvsdg::Region * region, size_t nbits)
{
  std::string s(nbits, 'D');
  return BitConstantOperation::create(region, BitValueRepresentation(s.c_str()));
}

}

#endif

/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/util/Hash.hpp>

namespace jlm::rvsdg
{

/* bistring type */

bittype::~bittype() noexcept
{}

std::string
bittype::debug_string() const
{
  return jlm::util::strfmt("bit", nbits());
}

bool
bittype::operator==(const jlm::rvsdg::Type & other) const noexcept
{
  auto type = dynamic_cast<const bittype *>(&other);
  return type != nullptr && this->nbits() == type->nbits();
}

std::size_t
bittype::ComputeHash() const noexcept
{
  auto typeHash = typeid(bittype).hash_code();
  auto numBitsHash = std::hash<size_t>()(nbits_);
  return util::CombineHashes(typeHash, numBitsHash);
}

std::shared_ptr<const bittype>
bittype::Create(std::size_t nbits)
{
  static const bittype static_instances[65] = {
    bittype(0),  bittype(1),  bittype(2),  bittype(3),  bittype(4),  bittype(5),  bittype(6),
    bittype(7),  bittype(8),  bittype(9),  bittype(10), bittype(11), bittype(12), bittype(13),
    bittype(14), bittype(15), bittype(16), bittype(17), bittype(18), bittype(19), bittype(20),
    bittype(21), bittype(22), bittype(23), bittype(24), bittype(25), bittype(26), bittype(27),
    bittype(28), bittype(29), bittype(30), bittype(31), bittype(32), bittype(33), bittype(34),
    bittype(35), bittype(36), bittype(37), bittype(38), bittype(39), bittype(40), bittype(41),
    bittype(42), bittype(43), bittype(44), bittype(45), bittype(46), bittype(47), bittype(48),
    bittype(49), bittype(50), bittype(51), bittype(52), bittype(53), bittype(54), bittype(55),
    bittype(56), bittype(57), bittype(58), bittype(59), bittype(60), bittype(61), bittype(62),
    bittype(63), bittype(64)
  };

  if (nbits <= 64)
  {
    if (nbits == 0)
    {
      throw util::Error("Number of bits must be greater than zero.");
    }

    return std::shared_ptr<const bittype>(std::shared_ptr<void>(), &static_instances[nbits]);
  }
  else
  {
    return std::make_shared<bittype>(nbits);
  }
}

}

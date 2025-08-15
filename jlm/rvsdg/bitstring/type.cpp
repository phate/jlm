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

BitType::~BitType() noexcept = default;

std::string
BitType::debug_string() const
{
  return jlm::util::strfmt("bit", nbits());
}

bool
BitType::operator==(const Type & other) const noexcept
{
  auto type = dynamic_cast<const BitType *>(&other);
  return type != nullptr && this->nbits() == type->nbits();
}

std::size_t
BitType::ComputeHash() const noexcept
{
  auto typeHash = typeid(BitType).hash_code();
  auto numBitsHash = std::hash<size_t>()(nbits_);
  return util::CombineHashes(typeHash, numBitsHash);
}

std::shared_ptr<const BitType>
BitType::Create(std::size_t nbits)
{
  static const BitType static_instances[65] = {
    BitType(0),  BitType(1),  BitType(2),  BitType(3),  BitType(4),  BitType(5),  BitType(6),
    BitType(7),  BitType(8),  BitType(9),  BitType(10), BitType(11), BitType(12), BitType(13),
    BitType(14), BitType(15), BitType(16), BitType(17), BitType(18), BitType(19), BitType(20),
    BitType(21), BitType(22), BitType(23), BitType(24), BitType(25), BitType(26), BitType(27),
    BitType(28), BitType(29), BitType(30), BitType(31), BitType(32), BitType(33), BitType(34),
    BitType(35), BitType(36), BitType(37), BitType(38), BitType(39), BitType(40), BitType(41),
    BitType(42), BitType(43), BitType(44), BitType(45), BitType(46), BitType(47), BitType(48),
    BitType(49), BitType(50), BitType(51), BitType(52), BitType(53), BitType(54), BitType(55),
    BitType(56), BitType(57), BitType(58), BitType(59), BitType(60), BitType(61), BitType(62),
    BitType(63), BitType(64)
  };

  if (nbits <= 64)
  {
    if (nbits == 0)
    {
      throw util::Error("Number of bits must be greater than zero.");
    }

    return std::shared_ptr<const BitType>(std::shared_ptr<void>(), &static_instances[nbits]);
  }
  else
  {
    return std::make_shared<BitType>(nbits);
  }
}

}

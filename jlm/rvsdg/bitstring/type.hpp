/*
 * Copyright 2011 2012 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_BITSTRING_TYPE_HPP
#define JLM_RVSDG_BITSTRING_TYPE_HPP

#include <jlm/rvsdg/type.hpp>
#include <jlm/util/common.hpp>

namespace jlm::rvsdg
{

class BitType final : public Type
{
public:
  ~BitType() noexcept override;

  explicit constexpr BitType(const size_t nbits)
      : nbits_(nbits)
  {}

  inline size_t
  nbits() const noexcept
  {
    return nbits_;
  }

  [[nodiscard]] std::string
  debug_string() const override;

  bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  TypeKind
  Kind() const noexcept override;

  /**
   * \brief Creates bit type of specified width
   *
   * \param nbits
   *    Width of type
   *
   * \returns
   *    Type representing bitstring of specified width.
   *
   * Returns an instance of a bitstring type with specified
   * width. Usually this returns a singleton object instance
   * for the type
   */
  static std::shared_ptr<const BitType>
  Create(std::size_t nbits);

private:
  size_t nbits_;
};

}

#endif

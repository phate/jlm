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

/* bitstring type */

class bittype final : public jlm::rvsdg::ValueType
{
public:
  virtual ~bittype() noexcept;

  inline constexpr bittype(size_t nbits)
      : nbits_(nbits)
  {}

  inline size_t
  nbits() const noexcept
  {
    return nbits_;
  }

  virtual std::string
  debug_string() const override;

  virtual bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

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
  static std::shared_ptr<const bittype>
  Create(std::size_t nbits);

private:
  size_t nbits_;
};

}

#endif

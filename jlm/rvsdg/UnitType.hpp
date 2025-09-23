/*
 * Copyright 2025 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_UNITTYPE_HPP
#define JLM_RVSDG_UNITTYPE_HPP

#include <jlm/rvsdg/type.hpp>

namespace jlm::rvsdg
{

/**
 * \brief Unit type (type carrying no information)
 *
 * Represents the "unit" type: A type which has only a single inhabitant
 * without content and carries no information. This is used in places where
 * a formal argument or result is needed, but no information is carried.
 *
 * This roughly corresponds to the "void" type in C/C++.
 */
class UnitType final : public Type
{
public:
  ~UnitType() noexcept override;

  std::string
  debug_string() const override;

  bool
  operator==(const Type & other) const noexcept override;

  std::size_t
  ComputeHash() const noexcept override;

  TypeKind
  Kind() const noexcept override;

  static std::shared_ptr<const UnitType>
  Create();
};

}

#endif

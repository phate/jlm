/*
 * Copyright 2010 2011 2012 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_SUBSTITUTION_HPP
#define JLM_RVSDG_SUBSTITUTION_HPP

#include <jlm/util/common.hpp>

#include <unordered_map>

namespace jlm::rvsdg
{

class output;
class Region;
class StructuralInput;

class SubstitutionMap final
{
public:
  bool
  contains(const output & original) const noexcept
  {
    return output_map_.find(&original) != output_map_.end();
  }

  bool
  contains(const Region & original) const noexcept
  {
    return region_map_.find(&original) != region_map_.end();
  }

  bool
  contains(const StructuralInput & original) const noexcept
  {
    return structinput_map_.find(&original) != structinput_map_.end();
  }

  output &
  lookup(const output & original) const
  {
    if (!contains(original))
      throw jlm::util::error("Output not in substitution map.");

    return *output_map_.find(&original)->second;
  }

  Region &
  lookup(const Region & original) const
  {
    if (!contains(original))
      throw jlm::util::error("Region not in substitution map.");

    return *region_map_.find(&original)->second;
  }

  [[nodiscard]] StructuralInput &
  lookup(const StructuralInput & original) const
  {
    if (!contains(original))
      throw jlm::util::error("Structural input not in substitution map.");

    return *structinput_map_.find(&original)->second;
  }

  inline jlm::rvsdg::output *
  lookup(const jlm::rvsdg::output * original) const noexcept
  {
    auto i = output_map_.find(original);
    return i != output_map_.end() ? i->second : nullptr;
  }

  [[nodiscard]] rvsdg::Region *
  lookup(const jlm::rvsdg::Region * original) const noexcept
  {
    auto i = region_map_.find(original);
    return i != region_map_.end() ? i->second : nullptr;
  }

  StructuralInput *
  lookup(const StructuralInput * original) const noexcept
  {
    auto i = structinput_map_.find(original);
    return i != structinput_map_.end() ? i->second : nullptr;
  }

  inline void
  insert(const jlm::rvsdg::output * original, jlm::rvsdg::output * substitute)
  {
    output_map_[original] = substitute;
  }

  inline void
  insert(const rvsdg::Region * original, rvsdg::Region * substitute)
  {
    region_map_[original] = substitute;
  }

  inline void
  insert(const StructuralInput * original, StructuralInput * substitute)
  {
    structinput_map_[original] = substitute;
  }

private:
  std::unordered_map<const rvsdg::Region *, rvsdg::Region *> region_map_;
  std::unordered_map<const jlm::rvsdg::output *, jlm::rvsdg::output *> output_map_;
  std::unordered_map<const StructuralInput *, StructuralInput *> structinput_map_;
};

}

#endif

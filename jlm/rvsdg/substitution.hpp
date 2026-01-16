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

class Output;

class SubstitutionMap final
{
public:
  bool
  contains(const Output & original) const noexcept
  {
    return outputMap_.find(&original) != outputMap_.end();
  }

  Output &
  lookup(const Output & original) const
  {
    if (!contains(original))
      throw util::Error("Output not in substitution map.");

    return *outputMap_.find(&original)->second;
  }

  Output *
  lookup(const Output * original) const noexcept
  {
    auto i = outputMap_.find(original);
    return i != outputMap_.end() ? i->second : nullptr;
  }

  void
  insert(const Output * original, Output * substitute)
  {
    outputMap_[original] = substitute;
  }

private:
  std::unordered_map<const Output *, Output *> outputMap_{};
};

}

#endif

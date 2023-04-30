/*
 * Copyright 2010 2011 2012 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_SUBSTITUTION_HPP
#define JLM_RVSDG_SUBSTITUTION_HPP

#include <jlm/util/common.hpp>

#include <unordered_map>

namespace jive {

class output;
class region;
class structural_input;

class substitution_map final {
public:
	bool
	contains(const output & original) const noexcept
	{
		return output_map_.find(&original) != output_map_.end();
	}

	bool
	contains(const region & original) const noexcept
	{
		return region_map_.find(&original) != region_map_.end();
	}

	bool
	contains(const structural_input & original) const noexcept
	{
		return structinput_map_.find(&original) != structinput_map_.end();
	}

	output &
	lookup(const output & original) const
	{
		if (!contains(original))
			throw compiler_error("Output not in substitution map.");

		return *output_map_.find(&original)->second;
	}

	region &
	lookup(const region & original) const
	{
		if (!contains(original))
			throw compiler_error("Region not in substitution map.");

		return *region_map_.find(&original)->second;
	}

	structural_input &
	lookup(const structural_input & original) const
	{
		if (!contains(original))
			throw compiler_error("Structural input not in substitution map.");

		return *structinput_map_.find(&original)->second;
	}

	inline jive::output *
	lookup(const jive::output * original) const noexcept
	{
		auto i = output_map_.find(original);
		return i != output_map_.end() ? i->second : nullptr;
	}

	inline jive::region *
	lookup(const jive::region * original) const noexcept
	{
		auto i = region_map_.find(original);
		return i != region_map_.end() ? i->second : nullptr;
	}

	inline jive::structural_input *
	lookup(const jive::structural_input * original) const noexcept
	{
		auto i = structinput_map_.find(original);
		return i != structinput_map_.end() ? i->second : nullptr;
	}

	inline void
	insert(const jive::output * original, jive::output * substitute)
	{
		output_map_[original] = substitute;
	}

	inline void
	insert(const jive::region * original, jive::region * substitute)
	{
		region_map_[original] = substitute;
	}

	inline void
	insert(const jive::structural_input * original, jive::structural_input * substitute)
	{
		structinput_map_[original] = substitute;
	}

private:
	std::unordered_map<const jive::region*, jive::region*> region_map_;
	std::unordered_map<const jive::output*, jive::output*> output_map_;
	std::unordered_map<const jive::structural_input*, jive::structural_input*> structinput_map_;
};

}

#endif

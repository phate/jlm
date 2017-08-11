/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_RVSDG_HPP
#define JLM_IR_RVSDG_HPP

#include <jive/vsdg/graph.h>

namespace jlm {

class rvsdg final {
public:
	inline
	rvsdg(const std::string & target_triple)
	: target_triple_(target_triple)
	{}

	rvsdg(const rvsdg &) = delete;

	rvsdg(rvsdg &&) = delete;

	rvsdg &
	operator=(const rvsdg &) = delete;

	rvsdg &
	operator=(rvsdg &&) = delete;

	inline jive::graph *
	graph() noexcept
	{
		return &graph_;
	}

	inline const jive::graph *
	graph() const noexcept
	{
		return &graph_;
	}

	inline const std::string &
	target_triple() const noexcept
	{
		return target_triple_;
	}

private:
	jive::graph graph_;
	std::string target_triple_;
};

}

#endif

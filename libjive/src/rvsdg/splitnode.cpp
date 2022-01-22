/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/common.hpp>
#include <jive/rvsdg/graph.hpp>
#include <jive/rvsdg/resource.hpp>
#include <jive/rvsdg/simple-node.hpp>
#include <jive/rvsdg/splitnode.hpp>

#include <string.h>

namespace jive {

split_op::~split_op() noexcept
{}

bool
split_op::operator==(const operation & other) const noexcept
{
	/* treat this operation a bit specially: state that any two
	 * splits are not the same to unconditionally make them exempt
	 * from CSE
	*/
	return this == &other;
}
std::string
split_op::debug_string() const
{
	return "SPLIT";
}

jive_unop_reduction_path_t
split_op::can_reduce_operand(const jive::output * arg) const noexcept
{
	return jive_unop_reduction_none;
}

jive::output *
split_op::reduce_operand(jive_unop_reduction_path_t path, jive::output * arg) const
{
	return nullptr;
}

std::unique_ptr<jive::operation>
split_op::copy() const
{
	return std::unique_ptr<jive::operation>(new split_op(*this));
}

}

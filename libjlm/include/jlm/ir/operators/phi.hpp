/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_PHI_HPP
#define JLM_IR_OPERATORS_PHI_HPP

#include <jive/rvsdg/phi.hpp>

/*
	FIXME: This should be defined in jive.
*/
static inline bool
is_phi_output(const jive::output * output)
{
	using namespace jive;

	return is<phi::operation>(node_output::node(output));
}

/*
	FIXME: This should be defined in jive.
*/
static inline bool
is_phi_cv(const jive::output * output)
{
	using namespace jive;

	auto a = dynamic_cast<const argument*>(output);
	return a
	    && is<phi::operation>(a->region()->node())
	    && a->input() != nullptr;
}

static inline bool
is_phi_recvar_argument(const jive::output * output)
{
	using namespace jive;

	auto a = dynamic_cast<const argument*>(output);
	return a
	    && is<phi::operation>(a->region()->node())
	    && a->input() == nullptr;
}

/*
	FIXME: This should be defined in jive.
*/
static inline jive::result *
phi_result(const jive::output * output)
{
	JLM_DEBUG_ASSERT(is_phi_output(output));
	auto result = jive::node_output::node(output)->region()->result(output->index());
	JLM_DEBUG_ASSERT(result->output() == output);
	return result;
}

#endif

/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_THETA_HPP
#define JLM_LLVM_IR_OPERATORS_THETA_HPP

#include <jlm/rvsdg/theta.hpp>

namespace jlm {

/*
	FIXME: This should be defined in jive.
*/
static inline const jive::argument *
is_theta_argument(const jive::output * output)
{
	using namespace jive;

	auto a = dynamic_cast<const jive::argument*>(output);
	if (a && is<theta_op>(a->region()->node()))
		return a;

	return nullptr;
}

static inline const jive::result *
is_theta_result(const jive::input * input)
{
	using namespace jive;

	auto r = dynamic_cast<const jive::result*>(input);
	if (r && is<theta_op>(r->region()->node()))
		return r;

	return nullptr;
}

/*
	FIXME: This function exists in jive, but is currently (2020-05-21) broken.
*/
static inline const jive::theta_output *
is_theta_output(const jive::output * output)
{
	return dynamic_cast<const jive::theta_output*>(output);
}

}

#endif

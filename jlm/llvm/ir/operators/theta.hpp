/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_THETA_HPP
#define JLM_LLVM_IR_OPERATORS_THETA_HPP

#include <jlm/rvsdg/theta.hpp>

namespace jlm {

/*
	FIXME: This should be defined in librvsdg.
*/
static inline const jlm::rvsdg::argument *
is_theta_argument(const jlm::rvsdg::output * output)
{
	using namespace jlm::rvsdg;

	auto a = dynamic_cast<const jlm::rvsdg::argument*>(output);
	if (a && is<theta_op>(a->region()->node()))
		return a;

	return nullptr;
}

static inline const jlm::rvsdg::result *
is_theta_result(const jlm::rvsdg::input * input)
{
	using namespace jlm::rvsdg;

	auto r = dynamic_cast<const jlm::rvsdg::result*>(input);
	if (r && is<theta_op>(r->region()->node()))
		return r;

	return nullptr;
}

/*
	FIXME: This function exists in librvsdg, but is currently (2020-05-21) broken.
*/
static inline const jlm::rvsdg::theta_output *
is_theta_output(const jlm::rvsdg::output * output)
{
	return dynamic_cast<const jlm::rvsdg::theta_output*>(output);
}

}

#endif

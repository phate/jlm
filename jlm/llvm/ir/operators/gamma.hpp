/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_GAMMA_HPP
#define JLM_LLVM_IR_OPERATORS_GAMMA_HPP

#include <jlm/rvsdg/gamma.hpp>

namespace jlm::llvm
{

/*
	FIXME: This should be defined in librvsdg.
*/
static inline const rvsdg::argument *
is_gamma_argument(const rvsdg::output * output)
{
	using namespace rvsdg;

	auto a = dynamic_cast<const rvsdg::argument*>(output);
	if (a && is<gamma_op>(a->region()->node()))
		return a;

	return nullptr;
}

/*
	FIXME: This function exists in librvsdg, but is currently (2020-05-19) broken.
*/
static inline const rvsdg::gamma_output *
is_gamma_output(const rvsdg::output * output)
{
	return dynamic_cast<const rvsdg::gamma_output*>(output);
}

/*
	FIXME: This should be defined in librvsdg.
*/
static inline const rvsdg::result *
is_gamma_result(const rvsdg::input * input)
{
	using namespace rvsdg;

	auto r = dynamic_cast<const result*>(input);
	if (r && is<gamma_op>(r->region()->node()))
		return r;

	return nullptr;
}

}

/*
	FIXME: This function should be defined in librvsdg.
*/
static inline jlm::rvsdg::output *
is_invariant(const jlm::rvsdg::gamma_output * output)
{
	auto argument = dynamic_cast<const jlm::rvsdg::argument*>(output->result(0)->origin());
	if (!argument)
		return nullptr;

	size_t n;
	auto origin = argument->input()->origin();
	for (n = 1; n < output->nresults(); n++) {
		auto argument = dynamic_cast<const jlm::rvsdg::argument*>(output->result(n)->origin());
		if (argument == nullptr || argument->input()->origin() != origin)
			break;
	}

	return n == output->nresults() ? origin : nullptr;
}

#endif

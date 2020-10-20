/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators.hpp>
#include <jlm/ir/rvsdg-module.hpp>

namespace jlm {

/* call operator */

call_op::~call_op()
{}

bool
call_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const call_op*>(&other);
	if (!op || op->narguments() != narguments() || op->nresults() != nresults())
		return false;

	for (size_t n = 0; n < narguments(); n++) {
		if (op->argument(n) != argument(n))
			return false;
	}

	for (size_t n = 0; n < nresults(); n++) {
		if (op->result(n) != result(n))
			return false;
	}

	return true;
}

std::string
call_op::debug_string() const
{
	return "CALL";
}

std::unique_ptr<jive::operation>
call_op::copy() const
{
	return std::unique_ptr<jive::operation>(new call_op(*this));
}

jive::output *
trace_function_input(const jive::simple_node & node)
{
	JLM_DEBUG_ASSERT(is<call_op>(&node));

	auto origin = node.input(0)->origin();

	while (true) {
		if (is_lambda_output(origin))
			return origin;

		if (is_lambda_cv(origin)) {
			auto argument = static_cast<const jive::argument*>(origin);
			origin = argument->input()->origin();
			continue;
		}

		/*
			FIXME: This lambda argument is dangerous.
			It overlaps with is_lambda_cv().
		*/
		if (is_lambda_argument(origin))
			return origin;

		if (is_import(origin))
			return origin;

		if (auto output = is_gamma_output(origin)) {
			if (auto new_origin = is_invariant(output)) {
				origin = new_origin;
				continue;
			}

			return origin;
		}

		if (auto argument = is_gamma_argument(origin)) {
			origin = argument->input()->origin();
			continue;
		}

		if (auto output = is_theta_output(origin)) {
			if (is_invariant(output)) {
				origin = output->input()->origin();
				continue;
			}

			return origin;
		}

		if (auto argument = is_theta_argument(origin)) {
			auto output = static_cast<const jive::theta_input*>(argument->input())->output();
			if (is_invariant(output)) {
				origin = argument->input()->origin();
				continue;
			}

			return origin;
		}

		if (is_phi_cv(origin)) {
			auto argument = static_cast<const jive::argument*>(origin);
			origin = argument->input()->origin();
			continue;
		}

		if (is_phi_recvar_argument(origin)) {
			auto argument = static_cast<const jive::argument*>(origin);
			/*
				FIXME: This assumes that all recursion variables where added before the dependencies. It
				would be better if we did not use the index for retrieving the result, but instead
				explicitly encoded it in an phi_argument.
			*/
			origin = argument->region()->result(argument->index())->origin();
			continue;
		}

		if (is_phi_output(origin)) {
			origin = phi_result(origin)->origin();
			continue;
		}

		JLM_ASSERT(0 && "We should have never reached this statement.");
	}
}

lambda_node *
is_direct_call(const jive::simple_node & node)
{
	if (!is<call_op>(&node))
		return nullptr;

	auto output = trace_function_input(node);

	if (is_lambda_output(output)) {
		JLM_DEBUG_ASSERT(is<lambda_op>(output->node()));
		return static_cast<lambda_node*>(output->node());
	}

	return nullptr;
}

}

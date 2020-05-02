/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/ir/operators/lambda.hpp>

static void
test_argument_iterators()
{
	using namespace jlm;

	valuetype vt;
	rvsdg_module rm(filepath(""), "", "");

	{
		jive::fcttype ft({&vt}, {&vt});
		lambda_op op(ft, "f", linkage::external_linkage);

		lambda_builder lb;
		auto arguments = lb.begin_lambda(rm.graph()->root(), op);
		auto lambda = lb.end_lambda({arguments[0]});

		std::vector<jive::argument*> args;
		for (auto it = lambda->begin_argument(); it != lambda->end_argument(); it++)
			args.push_back(it.argument());

		assert(args.size() == 1 && args[0] == lambda->subregion()->argument(0));
	}

	{
		jive::fcttype ft({}, {&vt});
		lambda_op op(ft, "f", linkage::external_linkage);

		lambda_builder lb;
		lb.begin_lambda(rm.graph()->root(), op);

		auto nullary = create_testop(lb.subregion(), {}, {&vt});

		auto lambda = lb.end_lambda(nullary);

		size_t narguments = 0;
		for (auto it = lambda->begin_argument(); it != lambda->end_argument(); it++)
			narguments++;

		assert(narguments == 0);
	}

	{
		auto i = rm.graph()->add_import({vt, ""});

		jive::fcttype ft({&vt, &vt, &vt}, {&vt, &vt});
		lambda_op op(ft, "f", linkage::external_linkage);

		lambda_builder lb;
		auto arguments = lb.begin_lambda(rm.graph()->root(), op);

		auto cv = lb.add_dependency(i);

		auto lambda = lb.end_lambda({arguments[0], cv});

		std::vector<jive::argument*> args;
		for (auto it = lambda->begin_argument(); it != lambda->end_argument(); it++)
			args.push_back(it.argument());

		assert(args.size() == 3);
		assert(args[0] == lambda->subregion()->argument(0));
		assert(args[1] == lambda->subregion()->argument(1));
		assert(args[2] == lambda->subregion()->argument(2));
	}
}

static int
test()
{
	test_argument_iterators();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/ir/operators/test-lambda", test)

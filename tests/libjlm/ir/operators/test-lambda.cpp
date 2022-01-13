/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/ir/RvsdgModule.hpp>
#include <jlm/ir/operators/lambda.hpp>

static void
test_argument_iterators()
{
	using namespace jlm;

	valuetype vt;
	rvsdg_module rm(filepath(""), "", "");

	{
		jive::fcttype ft({&vt}, {&vt});

		auto lambda = lambda::node::create(rm.graph()->root(), ft, "f", linkage::external_linkage, {});
		lambda->finalize({lambda->fctargument(0)});

		std::vector<jive::argument*> args;
		for (auto & argument : lambda->fctarguments())
			args.push_back(&argument);

		assert(args.size() == 1 && args[0] == lambda->fctargument(0));
	}

	{
		jive::fcttype ft({}, {&vt});

		auto lambda = lambda::node::create(rm.graph()->root(), ft, "f", linkage::external_linkage, {});

		auto nullary = create_testop(lambda->subregion(), {}, {&vt});

		lambda->finalize({nullary});

		assert(lambda->nfctarguments() == 0);
	}

	{
		auto i = rm.graph()->add_import({vt, ""});

		jive::fcttype ft({&vt, &vt, &vt}, {&vt, &vt});

		auto lambda = lambda::node::create(rm.graph()->root(), ft, "f", linkage::external_linkage, {});

		auto cv = lambda->add_ctxvar(i);

		lambda->finalize({lambda->fctargument(0), cv});

		std::vector<jive::argument*> args;
		for (auto & argument : lambda->fctarguments())
			args.push_back(&argument);

		assert(args.size() == 3);
		assert(args[0] == lambda->fctargument(0));
		assert(args[1] == lambda->fctargument(1));
		assert(args[2] == lambda->fctargument(2));
	}
}

static void
test_invalid_operand_region()
{
	using namespace jlm;

	valuetype vt;
	jive::fcttype fcttype({}, {&vt});

	auto module = rvsdg_module::create(filepath(""), "", "");
	auto graph = module->graph();

	auto fct1 = lambda::node::create(graph->root(), fcttype, "fct1", linkage::external_linkage);
	auto result = create_testop(graph->root(), {}, {&vt})[0];

	bool invalid_region_error_caught = false;
	try {
		fct1->finalize({result});
	} catch (jlm::error&) {
		invalid_region_error_caught = true;
	}

	assert(invalid_region_error_caught);
}

static int
test()
{
	test_argument_iterators();
	test_invalid_operand_region();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/ir/operators/test-lambda", test)

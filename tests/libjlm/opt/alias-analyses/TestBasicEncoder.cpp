/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "AliasAnalysesTests.hpp"

#include <test-registry.hpp>

#include <jive/view.hpp>

#include <jlm/opt/alias-analyses/BasicEncoder.hpp>
#include <jlm/opt/alias-analyses/Operators.hpp>
#include <jlm/opt/alias-analyses/pointsto-graph.hpp>
#include <jlm/opt/alias-analyses/steensgaard.hpp>
#include <jlm/util/stats.hpp>

#include <iostream>

static std::unique_ptr<jlm::aa::ptg>
run_steensgaard(jlm::rvsdg_module & module)
{
	using namespace jlm;

	aa::Steensgaard stgd;
	return stgd.Analyze(module);
}

template <class OP> static bool
is(
	const jive::node & node,
	size_t ninputs,
	size_t noutputs)
{
	return jive::is<OP>(&node)
	    && node.ninputs() == ninputs
	    && node.noutputs() == noutputs;
}

static void
test_store1()
{
	auto validate_rvsdg = [](const store_test1 & test)
	{
		using namespace jlm;

		assert(test.lambda->subregion()->nnodes() == 10);

		auto lambda_exit_mux = jive::node_output::node(test.lambda->fctresult(0)->origin());
		assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 5, 1));

		/* FIXME: We can do better */
	};

	store_test1 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
//	jive::view(test.graph().root(), stdout);
	validate_rvsdg(test);
}

static void
test_store2()
{
	auto validate_rvsdg = [](const store_test2 & test)
	{
		using namespace jlm;

		/* FIXME: write assertions */
	};

	store_test2 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
//	jive::view(test.graph().root(), stdout);
	validate_rvsdg(test);
}

static void
test_load1()
{
	auto ValidateRvsdg = [](const load_test1 & test)
	{
		/* FIXME: write assertions */
	};

	load_test1 test;
//	jive::view(test.graph()->root(), stdout);

	auto ptg = run_steensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
	ValidateRvsdg(test);
}

static void
test_load2()
{
	auto validate_rvsdg = [](const load_test2 & test)
	{
		using namespace jlm;

		/* FIXME: insert assertions */
	};

	load_test2 test;
	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
	std::cout << jlm::aa::ptg::to_dot(*ptg);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
	jive::view(test.graph().root(), stdout);
	validate_rvsdg(test);
}

static void
test_call1()
{
	auto validate_rvsdg = [](const call_test1 & test)
	{
		using namespace jlm;

		/* validate f */
		{
			auto lambda_exit_mux = jive::node_output::node(test.lambda_f->fctresult(1)->origin());
			assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 6, 1));

			/* FIXME: We can do better */
		}

		/* validate g */
		{
			auto lambda_exit_mux = jive::node_output::node(test.lambda_g->fctresult(1)->origin());
			assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 6, 1));

			/* FIXME: We can do better */
		}

		/* validate h */
		{
			/* FIXME: We can do better */
		}
	};

	call_test1 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
//	jive::view(test.graph().root(), stdout);
	validate_rvsdg(test);
}

static void
test_call2()
{
	auto validate_rvsdg = [](const call_test2 & test)
	{
		using namespace jlm;

		/* validate create function */
		{
			assert(test.lambda_create->subregion()->nnodes() == 6);

			auto lambda_exit_mux = jive::node_output::node(test.lambda_create->fctresult(1)->origin());
			assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 4, 1));

			/* FIXME: We can do better */
		}

		/* validate destroy function */
		{
			assert(test.lambda_destroy->subregion()->nnodes() == 4);

			auto lambda_exit_mux = jive::node_output::node(test.lambda_destroy->fctresult(0)->origin());
			assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 4, 1));

			/* FIXME: We can do better */
		}

		/* validate test function */
		{
			assert(test.lambda_test->subregion()->nnodes() == 16);

			/* FIXME We can do better */
		}
	};

	call_test2 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
//	jive::view(test.graph().root(), stdout);
	validate_rvsdg(test);
}

static void
test_indirect_call()
{
	auto validate_rvsdg = [](const indirect_call_test & test)
	{
		using namespace jlm;

		/* validate indcall function */
		{
			assert(test.lambda_indcall->subregion()->nnodes() == 5);

			auto lambda_exit_mux = jive::node_output::node(test.lambda_indcall->fctresult(1)->origin());
			assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 4, 1));

			auto call_exit_mux = jive::node_output::node(lambda_exit_mux->input(0)->origin());
			assert(is<aa::CallExitMemStateOperator>(*call_exit_mux, 1, 4));

			auto call = jive::node_output::node(call_exit_mux->input(0)->origin());
			assert(is<call_op>(*call, 3, 3));

			auto call_entry_mux = jive::node_output::node(call->input(1)->origin());
			assert(is<aa::CallEntryMemStateOperator>(*call_entry_mux, 4, 1));

			auto lambda_entry_mux = jive::node_output::node(call_entry_mux->input(1)->origin());
			assert(is<aa::LambdaEntryMemStateOperator>(*lambda_entry_mux, 1, 4));
		}

		/* validate test function */
		{
			assert(test.lambda_test->subregion()->nnodes() == 9);

			auto lambda_exit_mux = jive::node_output::node(test.lambda_test->fctresult(1)->origin());
			assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 4, 1));

			auto call_exit_mux = jive::node_output::node(lambda_exit_mux->input(0)->origin());
			assert(is<aa::CallExitMemStateOperator>(*call_exit_mux, 1, 4));

			auto call = jive::node_output::node(call_exit_mux->input(0)->origin());
			assert(is<call_op>(*call, 4, 3));

			auto call_entry_mux = jive::node_output::node(call->input(2)->origin());
			assert(is<aa::CallEntryMemStateOperator>(*call_entry_mux, 4, 1));

			call_exit_mux = jive::node_output::node(call_entry_mux->input(0)->origin());
			assert(is<aa::CallExitMemStateOperator>(*call_exit_mux, 1, 4));

			call = jive::node_output::node(call_exit_mux->input(0)->origin());
			assert(is<call_op>(*call, 4, 3));

			call_entry_mux = jive::node_output::node(call->input(2)->origin());
			assert(is<aa::CallEntryMemStateOperator>(*call_entry_mux, 4, 1));

			auto lambda_entry_mux = jive::node_output::node(call_entry_mux->input(1)->origin());
			assert(is<aa::LambdaEntryMemStateOperator>(*lambda_entry_mux, 1, 4));
		}
	};

	indirect_call_test test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
//	jive::view(test.graph().root(), stdout);
	validate_rvsdg(test);
}

static void
test_gamma()
{
	auto validate_rvsdg = [](const gamma_test & test)
	{
		/* FIXME: write assertions */
	};

	gamma_test test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
//	jive::view(test.graph().root(), stdout);
	validate_rvsdg(test);
}

static void
test_theta()
{
	auto validate_rvsdg = [](const theta_test & test)
	{
		using namespace jlm;

		assert(test.lambda->subregion()->nnodes() == 4);

		auto lambda_exit_mux = jive::node_output::node(test.lambda->fctresult(0)->origin());
		assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 1, 1));

		auto theta = jive::node_output::node(lambda_exit_mux->input(0)->origin());
		assert(theta == test.theta);

		auto store = jive::node_output::node(test.theta->output(4)->result()->origin());
		assert(is<store_op>(*store, 3, 1));
		assert(store->input(2)->origin() == test.theta->output(4)->argument());

		auto lambda_entry_mux = jive::node_output::node(test.theta->input(4)->origin());
		assert(is<aa::LambdaEntryMemStateOperator>(*lambda_entry_mux, 1, 1));
	};

	theta_test test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
//	jive::view(test.graph().root(), stdout);
	validate_rvsdg(test);
}

static void
test_delta1()
{
	auto ValidateRvsdg = [](const delta_test1 & test)
	{
		/* FIXME: write assertions */
	};

	delta_test1 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
//	jive::view(test.graph().root(), stdout);
	ValidateRvsdg(test);
}

static void
test_delta2()
{
	auto ValidateRvsdg = [](const delta_test2 & test)
	{
		/* FIXME: write assertions */
	};

	delta_test2 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
//	jive::view(test.graph().root(), stdout);
	ValidateRvsdg(test);
}

static void
test_imports()
{
	auto ValidateRvsdg = [](const import_test & test)
	{
		/* FIXME: write assertions */
	};

	import_test test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
//	std::cout << jlm::aa::ptr::to_dot(*ptg);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
//	jive::view(test.graph().root(), stdout);
	ValidateRvsdg(test);
}

static void
test_phi()
{
	auto validate_rvsdg = [](const phi_test & test)
	{
		using namespace jlm;

		/* FIXME: add assertions */
	};

	phi_test test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
//	jive::view(test.graph().root(), stdout);
	validate_rvsdg(test);
}

static int
test()
{
	test_store1();
	test_store2();

	test_load1();
	test_load2();

	test_call1();
	test_call2();

	test_indirect_call();

	test_gamma();
	test_theta();

	test_delta1();
	test_delta2();

	test_imports();

	test_phi();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/alias-analyses/TestBasicEncoder", test)

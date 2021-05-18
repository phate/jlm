/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "aa-tests.hpp"

#include <test-registry.hpp>

#include <jive/view.hpp>

#include <jlm/opt/alias-analyses/encoders.hpp>
#include <jlm/opt/alias-analyses/operators.hpp>
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
		assert(is<aa::lambda_aamux_op>(*lambda_exit_mux, 5, 1));

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

		auto & graph = *test.module().graph();

		assert(nnodes(graph.root()) == 12);

		auto mux = jive::node_output::node(test.lambda->subregion()->result(0)->origin());
		assert(jive::is<memstatemux_op>(mux));
		assert(mux->ninputs() == 5);

		assert(test.alloca_a->output(1)->nusers() == 1);
		assert(dynamic_cast<jive::simple_input*>(*test.alloca_a->output(1)->begin())->node() == mux);

		assert(test.alloca_b->output(1)->nusers() == 1);
		assert(dynamic_cast<jive::simple_input*>(*test.alloca_b->output(1)->begin())->node() == mux);

		assert(test.alloca_x->output(1)->nusers() == 1);
		auto store = dynamic_cast<jive::simple_input*>(*test.alloca_x->output(1)->begin())->node();
		assert(jive::is<store_op>(store) && store->noutputs() == 2);
		store = dynamic_cast<jive::simple_input*>(*store->output(0)->begin())->node();
		assert(jive::is<store_op>(store) && store->noutputs() == 2);
		assert(dynamic_cast<jive::simple_input*>(*store->output(0)->begin())->node() == mux);

		assert(test.alloca_y->output(1)->nusers() == 1);
		store = dynamic_cast<jive::simple_input*>(*test.alloca_y->output(1)->begin())->node();
		assert(jive::is<store_op>(store) && store->noutputs() == 2);
		store = dynamic_cast<jive::simple_input*>(*store->output(0)->begin())->node();
		assert(jive::is<store_op>(store) && store->noutputs() == 2);
		assert(dynamic_cast<jive::simple_input*>(*store->output(0)->begin())->node() == mux);

		assert(test.alloca_p->output(1)->nusers() == 1);
		store = dynamic_cast<jive::simple_input*>(*test.alloca_p->output(1)->begin())->node();
		assert(jive::is<store_op>(store) && store->noutputs() == 1);
		store = dynamic_cast<jive::simple_input*>(*store->output(0)->begin())->node();
		assert(jive::is<store_op>(store) && store->noutputs() == 1);
		assert(dynamic_cast<jive::simple_input*>(*store->output(0)->begin())->node() == mux);
	};

	store_test2 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);

	//jlm::aa::encode(*ptg, test.module());
//	jive::view(test.graph().root(), stdout);
	validate_rvsdg(test);
}

static void
test_load1()
{
	load_test1 test;
//	jive::view(test.graph()->root(), stdout);
	auto ptg = run_steensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);

	/* FIXME: validate RVSDG encoding of aa results */
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
			assert(is<aa::lambda_aamux_op>(*lambda_exit_mux, 6, 1));

			/* FIXME: We can do better */
		}

		/* validate g */
		{
			auto lambda_exit_mux = jive::node_output::node(test.lambda_g->fctresult(1)->origin());
			assert(is<aa::lambda_aamux_op>(*lambda_exit_mux, 6, 1));

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
			assert(is<aa::lambda_aamux_op>(*lambda_exit_mux, 4, 1));

			/* FIXME: We can do better */
		}

		/* validate destroy function */
		{
			assert(test.lambda_destroy->subregion()->nnodes() == 4);

			auto lambda_exit_mux = jive::node_output::node(test.lambda_destroy->fctresult(0)->origin());
			assert(is<aa::lambda_aamux_op>(*lambda_exit_mux, 4, 1));

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
			assert(is<aa::lambda_aamux_op>(*lambda_exit_mux, 4, 1));

			auto call_exit_mux = jive::node_output::node(lambda_exit_mux->input(0)->origin());
			assert(is<aa::call_aamux_op>(*call_exit_mux, 1, 4));

			auto call = jive::node_output::node(call_exit_mux->input(0)->origin());
			assert(is<call_op>(*call, 3, 3));

			auto call_entry_mux = jive::node_output::node(call->input(1)->origin());
			assert(is<aa::call_aamux_op>(*call_entry_mux, 4, 1));

			auto lambda_entry_mux = jive::node_output::node(call_entry_mux->input(1)->origin());
			assert(is<aa::lambda_aamux_op>(*lambda_entry_mux, 1, 4));
		}

		/* validate test function */
		{
			assert(test.lambda_test->subregion()->nnodes() == 9);

			auto lambda_exit_mux = jive::node_output::node(test.lambda_test->fctresult(1)->origin());
			assert(is<aa::lambda_aamux_op>(*lambda_exit_mux, 4, 1));

			auto call_exit_mux = jive::node_output::node(lambda_exit_mux->input(0)->origin());
			assert(is<aa::call_aamux_op>(*call_exit_mux, 1, 4));

			auto call = jive::node_output::node(call_exit_mux->input(0)->origin());
			assert(is<call_op>(*call, 4, 3));

			auto call_entry_mux = jive::node_output::node(call->input(2)->origin());
			assert(is<aa::call_aamux_op>(*call_entry_mux, 4, 1));

			call_exit_mux = jive::node_output::node(call_entry_mux->input(0)->origin());
			assert(is<aa::call_aamux_op>(*call_exit_mux, 1, 4));

			call = jive::node_output::node(call_exit_mux->input(0)->origin());
			assert(is<call_op>(*call, 4, 3));

			call_entry_mux = jive::node_output::node(call->input(2)->origin());
			assert(is<aa::call_aamux_op>(*call_entry_mux, 4, 1));

			auto lambda_entry_mux = jive::node_output::node(call_entry_mux->input(1)->origin());
			assert(is<aa::lambda_aamux_op>(*lambda_entry_mux, 1, 4));
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
		assert(is<aa::lambda_aamux_op>(*lambda_exit_mux, 1, 1));

		auto theta = jive::node_output::node(lambda_exit_mux->input(0)->origin());
		assert(theta == test.theta);

		auto store = jive::node_output::node(test.theta->output(4)->result()->origin());
		assert(is<store_op>(*store, 3, 1));
		assert(store->input(2)->origin() == test.theta->output(4)->argument());

		auto lambda_entry_mux = jive::node_output::node(test.theta->input(4)->origin());
		assert(is<aa::lambda_aamux_op>(*lambda_entry_mux, 1, 1));
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
	auto validate_rvsdg = [](const delta_test1 & test)
	{
		using namespace jlm;

		/* validate function g */
		{
			assert(test.lambda_g->subregion()->nnodes() == 3);

			auto mux = jive::node_output::node(test.lambda_g->fctresult(1)->origin());
			assert(is<memstatemux_op>(*mux, 1, 1));

			auto load = jive::node_output::node(mux->input(0)->origin());
			assert(jive::is<load_op>(load));
			assert(load->ninputs() == 2);

			mux = jive::node_output::node(load->input(1)->origin());
			assert(is<memstatemux_op>(*mux, 1, 1));
		}

		/* validate function h */
		{
			assert(test.lambda_h->subregion()->nnodes() == 7);

			auto mux = jive::node_output::node(test.lambda_h->fctresult(1)->origin());
			assert(is<memstatemux_op>(*mux, 1, 1));

			mux = jive::node_output::node(mux->input(0)->origin());
			assert(is<memstatemux_op>(*mux, 1, 1));

			auto call = jive::node_output::node(mux->input(0)->origin());
			assert(jive::is<call_op>(call));
			assert(call->ninputs() == 3);

			mux = jive::node_output::node(call->input(2)->origin());
			assert(is<memstatemux_op>(*mux, 1, 1));

			auto store = jive::node_output::node(mux->input(0)->origin());
			assert(jive::is<store_op>(store));
			assert(store->ninputs() == 3);

			mux = jive::node_output::node(store->input(2)->origin());
			assert(is<memstatemux_op>(*mux, 1, 1));
		}
	};

	delta_test1 test;
	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());

//	jlm::aa::encode(*ptg, test.module());
	jive::view(test.graph().root(), stdout);
	validate_rvsdg(test);
}

static void
test_delta2()
{
	auto validate_rvsdg = [](const delta_test2 & test)
	{
		using namespace jlm;

		/* validate function f1 */
		{
			assert(test.lambda_f1->subregion()->nnodes() == 4);

			auto lambda_exit_mux = jive::node_output::node(test.lambda_f1->fctresult(0)->origin());
			assert(is<memstatemux_op>(*lambda_exit_mux, 2, 1));

			auto store_d1 = jive::node_output::node(lambda_exit_mux->input(1)->origin());
			assert(is<store_op>(*store_d1, 3, 1));

			auto lambda_entry_mux = jive::node_output::node(store_d1->input(2)->origin());
			assert(is<memstatemux_op>(*lambda_entry_mux, 1, 2));

			assert(lambda_exit_mux->input(0)->origin() == lambda_entry_mux->output(0));
		}

		/* validate function f2 */
		{
			assert(test.lambda_f2->subregion()->nnodes() == 9);

			auto lambda_exit_mux = jive::node_output::node(test.lambda_f2->fctresult(0)->origin());
			assert(is<memstatemux_op>(*lambda_exit_mux, 2, 1));

			auto store_d2 = jive::node_output::node(lambda_exit_mux->input(0)->origin());
			assert(is<store_op>(*store_d2, 3, 1));

			auto call_exit_mux = jive::node_output::node(store_d2->input(2)->origin());
			assert(is<memstatemux_op>(*call_exit_mux, 1, 2));
			assert(lambda_exit_mux->input(1)->origin() == call_exit_mux->output(1));

			auto call_f1 = jive::node_output::node(call_exit_mux->input(0)->origin());
			assert(is<call_op>(*call_f1, 2, 1));

			auto call_entry_mux = jive::node_output::node(call_f1->input(1)->origin());
			assert(is<memstatemux_op>(*call_entry_mux, 2, 1));

			auto store_d1 = jive::node_output::node(call_entry_mux->input(1)->origin());
			assert(is<store_op>(*store_d1, 3, 1));

			auto lambda_entry_mux = jive::node_output::node(store_d1->input(2)->origin());
			assert(is<memstatemux_op>(*lambda_entry_mux, 1, 2));
			assert(call_entry_mux->input(0)->origin() == lambda_entry_mux->output(0));
		}
	};

	delta_test2 test;
	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());

//	jlm::aa::encode(*ptg, test.module());
	jive::view(test.graph().root(), stdout);
	validate_rvsdg(test);
}

static void
test_imports()
{
	auto validate_rvsdg = [](const import_test & test)
	{
		using namespace jlm;

		/* validate function f1 */
		{
			assert(test.lambda_f1->subregion()->nnodes() == 4);

			auto lambda_exit_mux = jive::node_output::node(test.lambda_f1->fctresult(0)->origin());
			assert(is<aa::lambda_aamux_op>(*lambda_exit_mux, 2, 1));

			auto store_d1 = jive::node_output::node(lambda_exit_mux->input(0)->origin());
			assert(is<store_op>(*store_d1, 3, 1));

			auto lambda_entry_mux = jive::node_output::node(store_d1->input(2)->origin());
			assert(is<aa::lambda_aamux_op>(*lambda_entry_mux, 1, 2));

			assert(lambda_exit_mux->input(1)->origin() == lambda_entry_mux->output(1));
		}

		/* validate function f2 */
		{
			assert(test.lambda_f2->subregion()->nnodes() == 9);

			auto lambda_exit_mux = jive::node_output::node(test.lambda_f2->fctresult(0)->origin());
			assert(is<aa::lambda_aamux_op>(*lambda_exit_mux, 2, 1));

			auto store_d2 = jive::node_output::node(lambda_exit_mux->input(1)->origin());
			assert(is<store_op>(*store_d2, 3, 1));

			auto call_exit_mux = jive::node_output::node(store_d2->input(2)->origin());
			assert(is<aa::call_aamux_op>(*call_exit_mux, 1, 2));
			assert(lambda_exit_mux->input(0)->origin() == call_exit_mux->output(0));

			auto call_f1 = jive::node_output::node(call_exit_mux->input(0)->origin());
			assert(is<call_op>(*call_f1, 2, 1));

			auto call_entry_mux = jive::node_output::node(call_f1->input(1)->origin());
			assert(is<aa::call_aamux_op>(*call_entry_mux, 2, 1));

			auto store_d1 = jive::node_output::node(call_entry_mux->input(0)->origin());
			assert(is<store_op>(*store_d1, 3, 1));

			auto lambda_entry_mux = jive::node_output::node(store_d1->input(2)->origin());
			assert(is<aa::lambda_aamux_op>(*lambda_entry_mux, 1, 2));
			assert(call_entry_mux->input(1)->origin() == lambda_entry_mux->output(1));
		}
	};

	import_test test;
	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());

//	jlm::aa::encode(*ptg, test.module());
	jive::view(test.graph().root(), stdout);
	validate_rvsdg(test);
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
//	test_store2();
//	test_load1();
	test_load2();

	test_call1();
	test_call2();

	test_indirect_call();

	test_gamma();
	test_theta();

//	test_delta1();
//	test_delta2();

//	test_imports();

	test_phi();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/alias-analyses/test-encoders", test)

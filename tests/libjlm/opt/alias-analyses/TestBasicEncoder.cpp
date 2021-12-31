/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "AliasAnalysesTests.hpp"

#include <test-registry.hpp>

#include <jive/view.hpp>

#include <jlm/opt/alias-analyses/BasicEncoder.hpp>
#include <jlm/opt/alias-analyses/Operators.hpp>
#include <jlm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/util/stats.hpp>

#include <iostream>

static std::unique_ptr<jlm::aa::PointsToGraph>
run_steensgaard(jlm::rvsdg_module & module)
{
	using namespace jlm;

	aa::Steensgaard stgd;
  StatisticsDescriptor sd;
	return stgd.Analyze(module, sd);
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
TestStore1()
{
	auto ValidateRvsdg = [](const StoreTest1 & test)
	{
		using namespace jlm;

		assert(test.lambda->subregion()->nnodes() == 10);

		auto lambdaExitMerge = jive::node_output::node(test.lambda->fctresult(0)->origin());
		assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 5, 1));

    assert(test.alloca_d->output(1)->nusers() == 1);
    assert(test.alloca_c->output(1)->nusers() == 1);
    assert(test.alloca_b->output(1)->nusers() == 1);
    assert(test.alloca_a->output(1)->nusers() == 1);

    assert(input_node((*test.alloca_d->output(1)->begin())) == lambdaExitMerge);

    auto storeD = input_node(*test.alloca_c->output(1)->begin());
    assert(is<store_op>(*storeD, 3, 1));
    assert(storeD->input(0)->origin() == test.alloca_c->output(0));
    assert(storeD->input(1)->origin() == test.alloca_d->output(0));

    auto storeC = input_node(*test.alloca_b->output(1)->begin());
    assert(is<store_op>(*storeC, 3, 1));
    assert(storeC->input(0)->origin() == test.alloca_b->output(0));
    assert(storeC->input(1)->origin() == test.alloca_c->output(0));

    auto storeB = input_node(*test.alloca_a->output(1)->begin());
    assert(is<store_op>(*storeB, 3, 1));
    assert(storeB->input(0)->origin() == test.alloca_a->output(0));
    assert(storeB->input(1)->origin() == test.alloca_b->output(0));
	};

	StoreTest1 test;
  // jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
	// std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
	// jive::view(test.graph().root(), stdout);
	ValidateRvsdg(test);
}

static void
TestStore2()
{
	auto ValidateRvsdg = [](const StoreTest2 & test)
	{
		using namespace jlm;

    assert(test.lambda->subregion()->nnodes() == 12);

    auto lambdaExitMerge = jive::node_output::node(test.lambda->fctresult(0)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 6, 1));

    assert(test.alloca_a->output(1)->nusers() == 1);
    assert(test.alloca_b->output(1)->nusers() == 1);
    assert(test.alloca_x->output(1)->nusers() == 1);
    assert(test.alloca_y->output(1)->nusers() == 1);
    assert(test.alloca_p->output(1)->nusers() == 1);

    assert(input_node((*test.alloca_a->output(1)->begin())) == lambdaExitMerge);
    assert(input_node((*test.alloca_b->output(1)->begin())) == lambdaExitMerge);

    auto storeA = input_node(*test.alloca_a->output(0)->begin());
    assert(is<store_op>(*storeA, 4, 2));
    assert(storeA->input(0)->origin() == test.alloca_x->output(0));

    auto storeB = input_node(*test.alloca_b->output(0)->begin());
    assert(is<store_op>(*storeB, 4, 2));
    assert(storeB->input(0)->origin() == test.alloca_y->output(0));
    assert(jive::node_output::node(storeB->input(2)->origin()) == storeA);
    assert(jive::node_output::node(storeB->input(3)->origin()) == storeA);

    auto storeX = input_node(*test.alloca_p->output(1)->begin());
    assert(is<store_op>(*storeX, 3, 1));
    assert(storeX->input(0)->origin() == test.alloca_p->output(0));
    assert(storeX->input(1)->origin() == test.alloca_x->output(0));

    auto storeY = input_node(*storeX->output(0)->begin());
    assert(is<store_op>(*storeY, 3, 1));
    assert(storeY->input(0)->origin() == test.alloca_p->output(0));
    assert(storeY->input(1)->origin() == test.alloca_y->output(0));
	};

	StoreTest2 test;
	// jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
	// std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
	// jive::view(test.graph().root(), stdout);
	ValidateRvsdg(test);
}

static void
TestLoad1()
{
	auto ValidateRvsdg = [](const LoadTest1 & test)
	{
    using namespace jlm;

    assert(test.lambda->subregion()->nnodes() == 4);

    auto lambdaExitMerge = jive::node_output::node(test.lambda->fctresult(1)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 1, 1));

    auto lambdaEntrySplit = input_node(*test.lambda->fctargument(1)->begin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 1));

    auto loadA = jive::node_output::node(test.lambda->fctresult(0)->origin());
    auto loadX = jive::node_output::node(loadA->input(0)->origin());

    assert(is<load_op>(*loadA, 2, 2));
    assert(loadA->input(1)->origin() == loadX->output(1));

    assert(is<load_op>(*loadX, 2, 2));
    assert(loadX->input(0)->origin() == test.lambda->fctargument(0));
    assert(loadX->input(1)->origin() == lambdaEntrySplit->output(0));
	};

	LoadTest1 test;
	// jive::view(test.graph()->root(), stdout);

	auto ptg = run_steensgaard(test.module());
	// std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
  // jive::view(test.graph().root(), stdout);
	ValidateRvsdg(test);
}

static void
TestLoad2()
{
	auto ValidateRvsdg = [](const LoadTest2 & test)
	{
		using namespace jlm;

    assert(test.lambda->subregion()->nnodes() == 14);

    auto lambdaExitMerge = jive::node_output::node(test.lambda->fctresult(0)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 6, 1));

    assert(test.alloca_a->output(1)->nusers() == 1);
    assert(test.alloca_b->output(1)->nusers() == 1);
    assert(test.alloca_x->output(1)->nusers() == 1);
    assert(test.alloca_y->output(1)->nusers() == 1);
    assert(test.alloca_p->output(1)->nusers() == 1);

    assert(input_node((*test.alloca_a->output(1)->begin())) == lambdaExitMerge);
    assert(input_node((*test.alloca_b->output(1)->begin())) == lambdaExitMerge);

    auto storeA = input_node(*test.alloca_a->output(0)->begin());
    assert(is<store_op>(*storeA, 4, 2));
    assert(storeA->input(0)->origin() == test.alloca_x->output(0));

    auto storeB = input_node(*test.alloca_b->output(0)->begin());
    assert(is<store_op>(*storeB, 4, 2));
    assert(storeB->input(0)->origin() == test.alloca_y->output(0));
    assert(jive::node_output::node(storeB->input(2)->origin()) == storeA);
    assert(jive::node_output::node(storeB->input(3)->origin()) == storeA);

    auto storeX = input_node(*test.alloca_x->output(0)->begin());
    assert(is<store_op>(*storeX, 3, 1));
    assert(storeX->input(0)->origin() == test.alloca_p->output(0));
    assert(storeX->input(1)->origin() == test.alloca_x->output(0));
    assert(storeX->input(2)->origin() == test.alloca_p->output(1));

    auto loadP = input_node(*storeX->output(0)->begin());
    assert(is<load_op>(*loadP, 2, 2));
    assert(loadP->input(0)->origin() == test.alloca_p->output(0));

    auto loadXY = input_node(*loadP->output(0)->begin());
    assert(is<load_op>(*loadXY, 3, 3));
    assert(jive::node_output::node(loadXY->input(1)->origin()) == storeB);
    assert(jive::node_output::node(loadXY->input(2)->origin()) == storeB);

    auto storeY = input_node(*loadXY->output(0)->begin());
    assert(is<store_op>(*storeY, 4, 2));
    assert(storeY->input(0)->origin() == test.alloca_y->output(0));
    assert(jive::node_output::node(storeY->input(2)->origin()) == loadXY);
    assert(jive::node_output::node(storeY->input(3)->origin()) == loadXY);
	};

	LoadTest2 test;
	// jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
	// std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
	// jive::view(test.graph().root(), stdout);
	ValidateRvsdg(test);
}

static void
TestCall1()
{
	auto ValidateRvsdg = [](const CallTest1 & test)
	{
		using namespace jlm;

		/* validate f */
		{
      auto lambdaEntrySplit = input_node(*test.lambda_f->fctargument(2)->begin());
			auto lambdaExitMerge = jive::node_output::node(test.lambda_f->fctresult(1)->origin());
      auto loadX = input_node(*test.lambda_f->fctargument(0)->begin());
      auto loadY = input_node(*test.lambda_f->fctargument(1)->begin());

      assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 6, 1));
      assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 6));

      assert(is<load_op>(*loadX, 2, 2));
      assert(jive::node_output::node(loadX->input(1)->origin()) == lambdaEntrySplit);

      assert(is<load_op>(*loadY, 2, 2));
      assert(jive::node_output::node(loadY->input(1)->origin()) == lambdaEntrySplit);
		}

		/* validate g */
		{
      auto lambdaEntrySplit = input_node(*test.lambda_g->fctargument(2)->begin());
      auto lambdaExitMerge = jive::node_output::node(test.lambda_g->fctresult(1)->origin());
      auto loadX = input_node(*test.lambda_g->fctargument(0)->begin());
      auto loadY = input_node(*test.lambda_g->fctargument(1)->begin());

      assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 6, 1));
      assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 6));

      assert(is<load_op>(*loadX, 2, 2));
      assert(jive::node_output::node(loadX->input(1)->origin()) == lambdaEntrySplit);

      assert(is<load_op>(*loadY, 2, 2));
      assert(jive::node_output::node(loadY->input(1)->origin()) == loadX);
		}

		/* validate h */
		{
			auto callEntryMerge = jive::node_output::node(test.callF->input(3)->origin());
      auto callExitSplit = input_node(*test.callF->output(1)->begin());

      assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 6, 1));
      assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 6));

      callEntryMerge = jive::node_output::node(test.callG->input(3)->origin());
      callExitSplit = input_node(*test.callG->output(1)->begin());

      assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 6, 1));
      assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 6));
		}
	};

	CallTest1 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
	// std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
	// jive::view(test.graph().root(), stdout);
	ValidateRvsdg(test);
}

static void
TestCall2()
{
	auto ValidateRvsdg = [](const CallTest2 & test)
	{
		using namespace jlm;

		/* validate create function */
		{
			assert(test.lambda_create->subregion()->nnodes() == 6);
		}

		/* validate destroy function */
		{
			assert(test.lambda_destroy->subregion()->nnodes() == 4);
		}

		/* validate test function */
		{
			assert(test.lambda_test->subregion()->nnodes() == 16);

      /* FIXME: The encoding is not completely correct. We are using a static heap model and the
       * two malloc allocations need to be merged as they are referring to the same abstract memory
       * location. Have a second look at this test.
       */
		}
	};

	CallTest2 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
	// std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
	// jive::view(test.graph().root(), stdout);
	ValidateRvsdg(test);
}

static void
TestIndirectCall()
{
	auto validate_rvsdg = [](const IndirectCallTest & test)
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

	IndirectCallTest test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
//	jive::view(test.graph().root(), stdout);
	validate_rvsdg(test);
}

static void
TestGamma()
{
	auto ValidateRvsdg = [](const GammaTest & test)
	{
    using namespace jlm;

    auto lambdaExitMerge = jive::node_output::node(test.lambda->fctresult(1)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 1, 1));

    auto loadTmp2 = jive::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<load_op>(*loadTmp2, 2, 2));

    auto loadTmp1 = jive::node_output::node(loadTmp2->input(1)->origin());
    assert(is<load_op>(*loadTmp1, 2, 2));

    auto gamma = jive::node_output::node(loadTmp1->input(1)->origin());
    assert(gamma == test.gamma);
	};

	GammaTest test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
	// std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
	// jive::view(test.graph().root(), stdout);
	ValidateRvsdg(test);
}

static void
TestTheta()
{
	auto ValidateRvsdg = [](const ThetaTest & test)
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

	ThetaTest test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
//	jive::view(test.graph().root(), stdout);
	ValidateRvsdg(test);
}

static void
TestDelta1()
{
	auto ValidateRvsdg = [](const DeltaTest1 & test)
	{
    using namespace jlm;

    assert(test.lambda_h->subregion()->nnodes() == 7);

    auto lambdaEntrySplit = input_node(*test.lambda_h->fctargument(0)->begin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 3));

    auto storeF = input_node(*test.constantFive->output(0)->begin());
    assert(is<store_op>(*storeF, 3, 1));
    assert(jive::node_output::node(storeF->input(2)->origin()) == lambdaEntrySplit);

    auto deltaStateIndex = storeF->input(2)->origin()->index();

    auto loadF = input_node(*test.lambda_g->fctargument(0)->begin());
    assert(is<load_op>(*loadF, 2, 2));
    assert(loadF->input(1)->origin()->index() == deltaStateIndex);
	};

	DeltaTest1 test;
  // jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
	// std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
	// jive::view(test.graph().root(), stdout);
	ValidateRvsdg(test);
}

static void
TestDelta2()
{
	auto ValidateRvsdg = [](const DeltaTest2 & test)
	{
    using namespace jlm;

    assert(test.lambda_f2->subregion()->nnodes() == 9);

    auto lambdaEntrySplit = input_node(*test.lambda_f2->fctargument(0)->begin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 4));

    auto storeD1InF2 = input_node(*test.lambda_f2->cvargument(0)->begin());
    assert(is<store_op>(*storeD1InF2, 3, 1));
    assert(jive::node_output::node(storeD1InF2->input(2)->origin()) == lambdaEntrySplit);

    auto d1StateIndex = storeD1InF2->input(2)->origin()->index();

    auto storeD1InF1 = input_node(*test.lambda_f1->cvargument(0)->begin());
    assert(is<store_op>(*storeD1InF1, 3, 1));

    assert(d1StateIndex == storeD1InF1->input(2)->origin()->index());

    auto storeD2InF2 = input_node(*test.lambda_f2->cvargument(1)->begin());
    assert(is<store_op>(*storeD1InF2, 3, 1));

    assert(d1StateIndex != storeD2InF2->input(2)->origin()->index());
	};

	DeltaTest2 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
	// std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
	// jive::view(test.graph().root(), stdout);
	ValidateRvsdg(test);
}

static void
TestImports()
{
	auto ValidateRvsdg = [](const ImportTest & test)
	{
    using namespace jlm;

    assert(test.lambda_f2->subregion()->nnodes() == 9);

    auto lambdaEntrySplit = input_node(*test.lambda_f2->fctargument(0)->begin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 4));

    auto storeD1InF2 = input_node(*test.lambda_f2->cvargument(0)->begin());
    assert(is<store_op>(*storeD1InF2, 3, 1));
    assert(jive::node_output::node(storeD1InF2->input(2)->origin()) == lambdaEntrySplit);

    auto d1StateIndex = storeD1InF2->input(2)->origin()->index();

    auto storeD1InF1 = input_node(*test.lambda_f1->cvargument(0)->begin());
    assert(is<store_op>(*storeD1InF1, 3, 1));

    assert(d1StateIndex == storeD1InF1->input(2)->origin()->index());

    auto storeD2InF2 = input_node(*test.lambda_f2->cvargument(1)->begin());
    assert(is<store_op>(*storeD1InF2, 3, 1));

    assert(d1StateIndex != storeD2InF2->input(2)->origin()->index());
	};

	ImportTest test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
	std::cout << jlm::aa::PointsToGraph::ToDot(*ptg);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
	jive::view(test.graph().root(), stdout);
	ValidateRvsdg(test);
}

static void
TestPhi()
{
	auto ValidateRvsdg = [](const PhiTest & test)
	{
		using namespace jlm;

		auto arrayStateIndex = (*test.alloca->output(1)->begin())->index();

    auto lambdaExitMerge = jive::node_output::node(test.lambda_fib->fctresult(0)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 3, 1));

    auto store = jive::node_output::node(lambdaExitMerge->input(arrayStateIndex)->origin());
    assert(is<store_op>(*store, 3, 1));

    auto gamma = jive::node_output::node(store->input(2)->origin());
    assert(gamma == test.gamma);

    auto gammaStateIndex = store->input(2)->origin()->index();

    auto load1 = jive::node_output::node(test.gamma->exitvar(gammaStateIndex)->result(0)->origin());
	  assert(is<load_op>(*load1, 2, 2));

    auto load2 = jive::node_output::node(load1->input(1)->origin());
    assert(is<load_op>(*load2, 2, 2));

    assert(load2->input(1)->origin()->index() == arrayStateIndex);
  };

	PhiTest test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = run_steensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

	jlm::aa::BasicEncoder::Encode(*ptg, test.module());
	// jive::view(test.graph().root(), stdout);
	ValidateRvsdg(test);
}

static int
test()
{
	TestStore1();
	TestStore2();

	TestLoad1();
	TestLoad2();

	TestCall1();
	TestCall2();
	TestIndirectCall();

	TestGamma();
	TestTheta();

	TestDelta1();
	TestDelta2();

	TestImports();

	TestPhi();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/alias-analyses/TestBasicEncoder", test)

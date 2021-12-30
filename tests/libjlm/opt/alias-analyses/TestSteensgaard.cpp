/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "AliasAnalysesTests.hpp"

#include <test-registry.hpp>

#include <jive/view.hpp>

#include <jlm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/util/stats.hpp>

#include <iostream>

static std::unique_ptr<jlm::aa::PointsToGraph>
runSteensgaard(jlm::rvsdg_module & module)
{
	using namespace jlm;

	aa::Steensgaard stgd;
  StatisticsDescriptor sd;
	return stgd.Analyze(module, sd);
}

static void
assertTargets(
	const jlm::aa::PointsToGraph::Node & node,
	const std::unordered_set<const jlm::aa::PointsToGraph::Node*> & targets)
{
	using namespace jlm::aa;

	assert(node.ntargets() == targets.size());

	std::unordered_set<const PointsToGraph::Node*> node_targets;
	for (auto & target : node.targets())
		node_targets.insert(&target);

	assert(targets == node_targets);
}

static void
TestStore1()
{
	auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const StoreTest1 & test)
	{
		assert(ptg.nallocnodes() == 5);
		assert(ptg.nregnodes() == 5);

		auto & alloca_a = ptg.GetAllocatorNode(test.alloca_a);
		auto & alloca_b = ptg.GetAllocatorNode(test.alloca_b);
		auto & alloca_c = ptg.GetAllocatorNode(test.alloca_c);
		auto & alloca_d = ptg.GetAllocatorNode(test.alloca_d);

		auto & palloca_a = ptg.GetRegisterNode(test.alloca_a->output(0));
		auto & palloca_b = ptg.GetRegisterNode(test.alloca_b->output(0));
		auto & palloca_c = ptg.GetRegisterNode(test.alloca_c->output(0));
		auto & palloca_d = ptg.GetRegisterNode(test.alloca_d->output(0));

		auto & lambda = ptg.GetAllocatorNode(test.lambda);
		auto & plambda = ptg.GetRegisterNode(test.lambda->output());

		assertTargets(alloca_a, {&alloca_b});
		assertTargets(alloca_b, {&alloca_c});
		assertTargets(alloca_c, {&alloca_d});
		assertTargets(alloca_d, {&ptg.memunknown()});

		assertTargets(palloca_a, {&alloca_a});
		assertTargets(palloca_b, {&alloca_b});
		assertTargets(palloca_c, {&alloca_c});
		assertTargets(palloca_d, {&alloca_d});

		assertTargets(lambda, {&ptg.memunknown()});
		assertTargets(plambda, {&lambda});
	};

	StoreTest1 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
	validate_ptg(*ptg, test);
}

static void
TestStore2()
{
	auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const StoreTest2 & test)
	{
		assert(ptg.nallocnodes() == 6);
		assert(ptg.nregnodes() == 6);

		auto & alloca_a = ptg.GetAllocatorNode(test.alloca_a);
		auto & alloca_b = ptg.GetAllocatorNode(test.alloca_b);
		auto & alloca_x = ptg.GetAllocatorNode(test.alloca_x);
		auto & alloca_y = ptg.GetAllocatorNode(test.alloca_y);
		auto & alloca_p = ptg.GetAllocatorNode(test.alloca_p);

		auto & palloca_a = ptg.GetRegisterNode(test.alloca_a->output(0));
		auto & palloca_b = ptg.GetRegisterNode(test.alloca_b->output(0));
		auto & palloca_x = ptg.GetRegisterNode(test.alloca_x->output(0));
		auto & palloca_y = ptg.GetRegisterNode(test.alloca_y->output(0));
		auto & palloca_p = ptg.GetRegisterNode(test.alloca_p->output(0));

		auto & lambda = ptg.GetAllocatorNode(test.lambda);
		auto & plambda = ptg.GetRegisterNode(test.lambda->output());

		assertTargets(alloca_a, {&ptg.memunknown()});
		assertTargets(alloca_b, {&ptg.memunknown()});
		assertTargets(alloca_x, {&alloca_a, &alloca_b});
		assertTargets(alloca_y, {&alloca_a, &alloca_b});
		assertTargets(alloca_p, {&alloca_x, &alloca_y});

		assertTargets(palloca_a, {&alloca_a, &alloca_b});
		assertTargets(palloca_b, {&alloca_a, &alloca_b});
		assertTargets(palloca_x, {&alloca_x, &alloca_y});
		assertTargets(palloca_y, {&alloca_x, &alloca_y});
		assertTargets(palloca_p, {&alloca_p});

		assertTargets(lambda, {&ptg.memunknown()});
		assertTargets(plambda, {&lambda});
	};

	StoreTest2 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
	validate_ptg(*ptg, test);
}

static void
TestLoad1()
{
	auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const LoadTest1 & test)
	{
		assert(ptg.nallocnodes() == 1);
		assert(ptg.nregnodes() == 3);

		auto & pload_p = ptg.GetRegisterNode(test.load_p->output(0));

		auto & lambda = ptg.GetAllocatorNode(test.lambda);
		auto & plambda = ptg.GetRegisterNode(test.lambda->output());
		auto & lambarg0 = ptg.GetRegisterNode(test.lambda->subregion()->argument(0));

		assertTargets(pload_p, {&ptg.memunknown()});

		assertTargets(plambda, {&lambda});
		assertTargets(lambarg0, {&ptg.memunknown()});
	};

	LoadTest1 test;
//	jive::view(test.graph()->root(), stdout);
	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

	validate_ptg(*ptg, test);
}

static void
TestLoad2()
{
	auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const LoadTest2 & test)
	{
		assert(ptg.nallocnodes() == 6);
		assert(ptg.nregnodes() == 8);

		/*
			We only care about the loads in this test, skipping the validation
			for all other nodes.
		*/
		auto & alloca_a = ptg.GetAllocatorNode(test.alloca_a);
		auto & alloca_b = ptg.GetAllocatorNode(test.alloca_b);
		auto & alloca_x = ptg.GetAllocatorNode(test.alloca_x);
		auto & alloca_y = ptg.GetAllocatorNode(test.alloca_y);

		auto & pload_x = ptg.GetRegisterNode(test.load_x->output(0));
		auto & pload_a = ptg.GetRegisterNode(test.load_a->output(0));

		assertTargets(pload_x, {&alloca_x, &alloca_y});
		assertTargets(pload_a, {&alloca_a, &alloca_b});
	};

	LoadTest2 test;
//	jive::view(test.graph()->root(), stdout);
	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

	validate_ptg(*ptg, test);
}

static void
TestGetElementPtr()
{
	auto validatePtg = [](const jlm::aa::PointsToGraph & ptg, const GetElementPtrTest & test)
	{
		assert(ptg.nallocnodes() == 1);
		assert(ptg.nregnodes() == 4);

		/*
			We only care about the getelemenptr's in this test, skipping the validation
			for all other nodes.
		*/
		auto & gepX = ptg.GetRegisterNode(test.getElementPtrX->output(0));
		auto & gepY = ptg.GetRegisterNode(test.getElementPtrY->output(0));

		assertTargets(gepX, {&ptg.memunknown()});
		assertTargets(gepY, {&ptg.memunknown()});
	};

	GetElementPtrTest test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
	validatePtg(*ptg, test);
}

static void
TestBitCast()
{
	auto validatePtg = [](const jlm::aa::PointsToGraph & ptg, const BitCastTest & test)
	{
		assert(ptg.nallocnodes() == 1);
		assert(ptg.nregnodes() == 3);

		auto & lambda = ptg.GetAllocatorNode(test.lambda);
		auto & lambdaOut = ptg.GetRegisterNode(test.lambda->output());
		auto & lambdaArg = ptg.GetRegisterNode(test.lambda->fctargument(0));

		auto & bitCast = ptg.GetRegisterNode(test.bitCast->output(0));

		assertTargets(lambdaOut, {&lambda});
		assertTargets(lambdaArg, {&ptg.memunknown()});
		assertTargets(bitCast, {&ptg.memunknown()});
	};

	BitCastTest test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
	validatePtg(*ptg, test);
}

static void
TestConstantPointerNull()
{
	auto validatePtg = [](const jlm::aa::PointsToGraph & ptg, const ConstantPointerNullTest & test)
	{
		assert(ptg.nallocnodes() == 1);
		assert(ptg.nregnodes() == 3);

		auto & lambda = ptg.GetAllocatorNode(test.lambda);
		auto & lambdaOut = ptg.GetRegisterNode(test.lambda->output());
		auto & lambdaArg = ptg.GetRegisterNode(test.lambda->fctargument(0));

		auto & null = ptg.GetRegisterNode(test.null->output(0));

		assertTargets(lambdaOut, {&lambda});
		assertTargets(lambdaArg, {&ptg.memunknown()});
		assertTargets(null, {&ptg.memunknown()});
	};

	ConstantPointerNullTest test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
	validatePtg(*ptg, test);
}

static void
TestBits2Ptr()
{
	auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const Bits2PtrTest & test)
	{
		assert(ptg.nallocnodes() == 2);
		assert(ptg.nregnodes() == 5);

		auto & call_out0 = ptg.GetRegisterNode(test.call->output(0));
		assertTargets(call_out0, {&ptg.memunknown()});

		auto & bits2ptr = ptg.GetRegisterNode(test.call->output(0));
		assertTargets(bits2ptr, {&ptg.memunknown()});
	};

	Bits2PtrTest test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
	validate_ptg(*ptg, test);
}

static void
TestCall1()
{
	auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const CallTest1 & test)
	{
		assert(ptg.nallocnodes() == 6);
		assert(ptg.nregnodes() == 12);

		auto & alloca_x = ptg.GetAllocatorNode(test.alloca_x);
		auto & alloca_y = ptg.GetAllocatorNode(test.alloca_y);
		auto & alloca_z = ptg.GetAllocatorNode(test.alloca_z);

		auto & palloca_x = ptg.GetRegisterNode(test.alloca_x->output(0));
		auto & palloca_y = ptg.GetRegisterNode(test.alloca_y->output(0));
		auto & palloca_z = ptg.GetRegisterNode(test.alloca_z->output(0));

		auto & lambda_f = ptg.GetAllocatorNode(test.lambda_f);
		auto & lambda_g = ptg.GetAllocatorNode(test.lambda_g);
		auto & lambda_h = ptg.GetAllocatorNode(test.lambda_h);

		auto & plambda_f = ptg.GetRegisterNode(test.lambda_f->output());
		auto & plambda_g = ptg.GetRegisterNode(test.lambda_g->output());
		auto & plambda_h = ptg.GetRegisterNode(test.lambda_h->output());

		auto & lambda_f_arg0 = ptg.GetRegisterNode(test.lambda_f->subregion()->argument(0));
		auto & lambda_f_arg1 = ptg.GetRegisterNode(test.lambda_f->subregion()->argument(1));

		auto & lambda_g_arg0 = ptg.GetRegisterNode(test.lambda_g->subregion()->argument(0));
		auto & lambda_g_arg1 = ptg.GetRegisterNode(test.lambda_g->subregion()->argument(1));

		auto & lambda_h_cv0 = ptg.GetRegisterNode(test.lambda_h->subregion()->argument(1));
		auto & lambda_h_cv1 = ptg.GetRegisterNode(test.lambda_h->subregion()->argument(2));

		assertTargets(palloca_x, {&alloca_x});
		assertTargets(palloca_y, {&alloca_y});
		assertTargets(palloca_z, {&alloca_z});

		assertTargets(plambda_f, {&lambda_f});
		assertTargets(plambda_g, {&lambda_g});
		assertTargets(plambda_h, {&lambda_h});

		assertTargets(lambda_f_arg0, {&alloca_x});
		assertTargets(lambda_f_arg1, {&alloca_y});

		assertTargets(lambda_g_arg0, {&alloca_z});
		assertTargets(lambda_g_arg1, {&alloca_z});

		assertTargets(lambda_h_cv0, {&lambda_f});
		assertTargets(lambda_h_cv1, {&lambda_g});
	};

	CallTest1 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
	validate_ptg(*ptg, test);
}

static void
TestCall2()
{
	auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const CallTest2 & test)
	{
		assert(ptg.nallocnodes() == 4);
		assert(ptg.nimpnodes() == 0);
		assert(ptg.nregnodes() == 11);

		auto & lambda_create = ptg.GetAllocatorNode(test.lambda_create);
		auto & lambda_create_out = ptg.GetRegisterNode(test.lambda_create->output());

		auto & lambda_destroy = ptg.GetAllocatorNode(test.lambda_destroy);
		auto & lambda_destroy_out = ptg.GetRegisterNode(test.lambda_destroy->output());
		auto & lambda_destroy_arg = ptg.GetRegisterNode(test.lambda_destroy->fctargument(0));

		auto & lambda_test = ptg.GetAllocatorNode(test.lambda_test);
		auto & lambda_test_out = ptg.GetRegisterNode(test.lambda_test->output());
		auto & lambda_test_cv1 = ptg.GetRegisterNode(test.lambda_test->cvargument(0));
		auto & lambda_test_cv2 = ptg.GetRegisterNode(test.lambda_test->cvargument(1));

		auto & call_create1_out = ptg.GetRegisterNode(test.call_create1->output(0));
		auto & call_create2_out = ptg.GetRegisterNode(test.call_create2->output(0));

		auto & malloc = ptg.GetAllocatorNode(test.malloc);
		auto & malloc_out = ptg.GetRegisterNode(test.malloc->output(0));

		assertTargets(lambda_create_out, {&lambda_create});

		assertTargets(lambda_destroy_out, {&lambda_destroy});
		assertTargets(lambda_destroy_arg, {&malloc});

		assertTargets(lambda_test_out, {&lambda_test});
		assertTargets(lambda_test_cv1, {&lambda_create});
		assertTargets(lambda_test_cv2, {&lambda_destroy});

		assertTargets(call_create1_out, {&malloc});
		assertTargets(call_create2_out, {&malloc});

		assertTargets(malloc_out, {&malloc});
	};

	CallTest2 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
	validate_ptg(*ptg, test);
}

static void
TestIndirectCall()
{
	auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const IndirectCallTest & test)
	{
		assert(ptg.nallocnodes() == 4);
		assert(ptg.nimpnodes() == 0);
		assert(ptg.nregnodes() == 8);

		auto & lambda_three = ptg.GetAllocatorNode(test.lambda_three);
		auto & lambda_three_out = ptg.GetRegisterNode(test.lambda_three->output());

		auto & lambda_four = ptg.GetAllocatorNode(test.lambda_four);
		auto & lambda_four_out = ptg.GetRegisterNode(test.lambda_four->output());

		auto & lambda_indcall = ptg.GetAllocatorNode(test.lambda_indcall);
		auto & lambda_indcall_out = ptg.GetRegisterNode(test.lambda_indcall->output());
		auto & lambda_indcall_arg = ptg.GetRegisterNode(test.lambda_indcall->fctargument(0));

		auto & lambda_test = ptg.GetAllocatorNode(test.lambda_test);
		auto & lambda_test_out = ptg.GetRegisterNode(test.lambda_test->output());
		auto & lambda_test_cv0 = ptg.GetRegisterNode(test.lambda_test->cvargument(0));
		auto & lambda_test_cv1 = ptg.GetRegisterNode(test.lambda_test->cvargument(1));
		auto & lambda_test_cv2 = ptg.GetRegisterNode(test.lambda_test->cvargument(2));

		assertTargets(lambda_three_out, {&lambda_three, &lambda_four});

		assertTargets(lambda_four_out, {&lambda_three, &lambda_four});

		assertTargets(lambda_indcall_out, {&lambda_indcall});
		assertTargets(lambda_indcall_arg, {&lambda_three, &lambda_four});

		assertTargets(lambda_test_out, {&lambda_test});
		assertTargets(lambda_test_cv0, {&lambda_indcall});
		assertTargets(lambda_test_cv1, {&lambda_three, &lambda_four});
		assertTargets(lambda_test_cv2, {&lambda_three, &lambda_four});
	};

	IndirectCallTest test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
	validate_ptg(*ptg, test);
}

static void
TestGamma()
{
	auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const GammaTest & test)
	{
		assert(ptg.nallocnodes() == 1);
		assert(ptg.nregnodes() == 15);

		for (size_t n = 1; n < 5; n++) {
			auto & lambda_arg = ptg.GetRegisterNode(test.lambda->fctargument(n));
			assertTargets(lambda_arg, {&ptg.memunknown()});
		}

		for (size_t n = 0; n < 4; n++) {
			auto & argument0 = ptg.GetRegisterNode(test.gamma->entryvar(n)->argument(0));
			auto & argument1 = ptg.GetRegisterNode(test.gamma->entryvar(n)->argument(1));

			assertTargets(argument0, {&ptg.memunknown()});
			assertTargets(argument1, {&ptg.memunknown()});
		}

		for (size_t n = 0; n < 4; n++) {
			auto & output = ptg.GetRegisterNode(test.gamma->exitvar(0));
			assertTargets(output, {&ptg.memunknown()});
		}
	};

	GammaTest test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
	validate_ptg(*ptg, test);
}

static void
TestTheta()
{
	auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const ThetaTest & test)
	{
		assert(ptg.nallocnodes() == 1);
		assert(ptg.nregnodes() == 5);

		auto & lambda = ptg.GetAllocatorNode(test.lambda);
		auto & lambda_arg1 = ptg.GetRegisterNode(test.lambda->fctargument(1));
		auto & lambda_out = ptg.GetRegisterNode(test.lambda->output());

		auto & gep_out = ptg.GetRegisterNode(test.gep->output(0));

		auto & theta_lv2_arg = ptg.GetRegisterNode(test.theta->output(2)->argument());
		auto & theta_lv2_out = ptg.GetRegisterNode(test.theta->output(2));

		assertTargets(lambda_arg1, {&ptg.memunknown()});
		assertTargets(lambda_out, {&lambda});

		assertTargets(gep_out, {&ptg.memunknown()});

		assertTargets(theta_lv2_arg, {&ptg.memunknown()});
		assertTargets(theta_lv2_out, {&ptg.memunknown()});
	};

	ThetaTest test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
	validate_ptg(*ptg, test);
}

static void
TestDelta1()
{
	auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const DeltaTest1 & test)
	{
		assert(ptg.nallocnodes() == 3);
		assert(ptg.nregnodes() == 6);

		auto & delta_f = ptg.GetAllocatorNode(test.delta_f);
		auto & pdelta_f = ptg.GetRegisterNode(test.delta_f->output());

		auto & lambda_g = ptg.GetAllocatorNode(test.lambda_g);
		auto & plambda_g = ptg.GetRegisterNode(test.lambda_g->output());
		auto & lambda_g_arg0 = ptg.GetRegisterNode(test.lambda_g->fctargument(0));

		auto & lambda_h = ptg.GetAllocatorNode(test.lambda_h);
		auto & plambda_h = ptg.GetRegisterNode(test.lambda_h->output());
		auto & lambda_h_cv0 = ptg.GetRegisterNode(test.lambda_h->cvargument(0));
		auto & lambda_h_cv1 = ptg.GetRegisterNode(test.lambda_h->cvargument(1));

		assertTargets(pdelta_f, {&delta_f});

		assertTargets(plambda_g, {&lambda_g});
		assertTargets(plambda_h, {&lambda_h});

		assertTargets(lambda_g_arg0, {&delta_f});

		assertTargets(lambda_h_cv0, {&delta_f});
		assertTargets(lambda_h_cv1, {&lambda_g});
	};

	DeltaTest1 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
	validate_ptg(*ptg, test);
}

static void
TestDelta2()
{
	auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const DeltaTest2 & test)
	{
		assert(ptg.nallocnodes() == 4);
		assert(ptg.nregnodes() == 8);

		auto & delta_d1 = ptg.GetAllocatorNode(test.delta_d1);
		auto & delta_d1_out = ptg.GetRegisterNode(test.delta_d1->output());

		auto & delta_d2 = ptg.GetAllocatorNode(test.delta_d2);
		auto & delta_d2_out = ptg.GetRegisterNode(test.delta_d2->output());

		auto & lambda_f1 = ptg.GetAllocatorNode(test.lambda_f1);
		auto & lambda_f1_out = ptg.GetRegisterNode(test.lambda_f1->output());
		auto & lambda_f1_cvd1 = ptg.GetRegisterNode(test.lambda_f1->cvargument(0));

		auto & lambda_f2 = ptg.GetAllocatorNode(test.lambda_f2);
		auto & lambda_f2_out = ptg.GetRegisterNode(test.lambda_f2->output());
		auto & lambda_f2_cvd1 = ptg.GetRegisterNode(test.lambda_f2->cvargument(0));
		auto & lambda_f2_cvd2 = ptg.GetRegisterNode(test.lambda_f2->cvargument(1));
		auto & lambda_f2_cvf1 = ptg.GetRegisterNode(test.lambda_f2->cvargument(2));

		assertTargets(delta_d1_out, {&delta_d1});
		assertTargets(delta_d2_out, {&delta_d2});

		assertTargets(lambda_f1_out, {&lambda_f1});
		assertTargets(lambda_f1_cvd1, {&delta_d1});

		assertTargets(lambda_f2_out, {&lambda_f2});
		assertTargets(lambda_f2_cvd1, {&delta_d1});
		assertTargets(lambda_f2_cvd2, {&delta_d2});
		assertTargets(lambda_f2_cvf1, {&lambda_f1});
	};

	DeltaTest2 test;
	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
	std::cout << jlm::aa::PointsToGraph::ToDot(*ptg);
	validate_ptg(*ptg, test);
}

static void
TestImports()
{
	auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const ImportTest & test)
	{
		assert(ptg.nallocnodes() == 2);
		assert(ptg.nimpnodes() == 2);
		assert(ptg.nregnodes() == 8);

		auto & d1 = ptg.GetImportNode(test.import_d1);
		auto & import_d1 = ptg.GetRegisterNode(test.import_d1);

		auto & d2 = ptg.GetImportNode(test.import_d2);
		auto & import_d2 = ptg.GetRegisterNode(test.import_d2);

		auto & lambda_f1 = ptg.GetAllocatorNode(test.lambda_f1);
		auto & lambda_f1_out = ptg.GetRegisterNode(test.lambda_f1->output());
		auto & lambda_f1_cvd1 = ptg.GetRegisterNode(test.lambda_f1->cvargument(0));

		auto & lambda_f2 = ptg.GetAllocatorNode(test.lambda_f2);
		auto & lambda_f2_out = ptg.GetRegisterNode(test.lambda_f2->output());
		auto & lambda_f2_cvd1 = ptg.GetRegisterNode(test.lambda_f2->cvargument(0));
		auto & lambda_f2_cvd2 = ptg.GetRegisterNode(test.lambda_f2->cvargument(1));
		auto & lambda_f2_cvf1 = ptg.GetRegisterNode(test.lambda_f2->cvargument(2));

		assertTargets(import_d1, {&d1});
		assertTargets(import_d2, {&d2});

		assertTargets(lambda_f1_out, {&lambda_f1});
		assertTargets(lambda_f1_cvd1, {&d1});

		assertTargets(lambda_f2_out, {&lambda_f2});
		assertTargets(lambda_f2_cvd1, {&d1});
		assertTargets(lambda_f2_cvd2, {&d2});
		assertTargets(lambda_f2_cvf1, {&lambda_f1});
	};

	ImportTest test;
	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
	std::cout << jlm::aa::PointsToGraph::ToDot(*ptg);
	validate_ptg(*ptg, test);
}

static void
TestPhi()
{
	auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const PhiTest & test)
	{
		assert(ptg.nallocnodes() == 3);
		assert(ptg.nregnodes() == 16);

		auto & lambda_fib = ptg.GetAllocatorNode(test.lambda_fib);
		auto & lambda_fib_out = ptg.GetRegisterNode(test.lambda_fib->output());
		auto & lambda_fib_arg1 = ptg.GetRegisterNode(test.lambda_fib->fctargument(1));

		auto & lambda_test = ptg.GetAllocatorNode(test.lambda_test);
		auto & lambda_test_out = ptg.GetRegisterNode(test.lambda_test->output());

		auto & phi_rv = ptg.GetRegisterNode(test.phi->begin_rv().output());
		auto & phi_rv_arg = ptg.GetRegisterNode(test.phi->begin_rv().output()->argument());

		auto & gamma_result = ptg.GetRegisterNode(test.gamma->subregion(0)->argument(1));
		auto & gamma_fib = ptg.GetRegisterNode(test.gamma->subregion(0)->argument(2));

		auto & alloca = ptg.GetAllocatorNode(test.alloca);
		auto & alloca_out = ptg.GetRegisterNode(test.alloca->output(0));

		assertTargets(lambda_fib_out, {&lambda_fib});
		assertTargets(lambda_fib_arg1, {&alloca});

		assertTargets(lambda_test_out, {&lambda_test});

		assertTargets(phi_rv, {&lambda_fib});
		assertTargets(phi_rv_arg, {&lambda_fib});

		assertTargets(gamma_result, {&alloca});
		assertTargets(gamma_fib, {&lambda_fib});

		assertTargets(alloca_out, {&alloca});
	};

	PhiTest test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
	validate_ptg(*ptg, test);
}

static int
test()
{
	TestStore1();
	TestStore2();

	TestLoad1();
	TestLoad2();

	TestGetElementPtr();

	TestBitCast();
	TestBits2Ptr();

	TestConstantPointerNull();

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

JLM_UNIT_TEST_REGISTER("libjlm/opt/alias-analyses/TestSteensgaard", test)

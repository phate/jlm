/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "aa-tests.hpp"

#include <test-registry.hpp>

#include <jive/view.hpp>

#include <jlm/opt/alias-analyses/operators.hpp>
#include <jlm/opt/alias-analyses/pointsto-graph.hpp>
#include <jlm/opt/alias-analyses/steensgaard.hpp>
#include <jlm/util/stats.hpp>

#include <iostream>

static std::unique_ptr<jlm::aa::ptg>
runSteensgaard(jlm::rvsdg_module & module)
{
	using namespace jlm;

	aa::Steensgaard stgd;
	return stgd.Analyze(module);
}

static void
assertTargets(
	const jlm::aa::ptg::node & node,
	const std::unordered_set<const jlm::aa::ptg::node*> & targets)
{
	using namespace jlm::aa;

	assert(node.ntargets() == targets.size());

	std::unordered_set<const ptg::node*> node_targets;
	for (auto & target : node.targets())
		node_targets.insert(&target);

	assert(targets == node_targets);
}

static void
TestStore1()
{
	auto validate_ptg = [](const jlm::aa::ptg & ptg, const store_test1 & test)
	{
		assert(ptg.nallocnodes() == 5);
		assert(ptg.nregnodes() == 5);

		auto & alloca_a = ptg.find(test.alloca_a);
		auto & alloca_b = ptg.find(test.alloca_b);
		auto & alloca_c = ptg.find(test.alloca_c);
		auto & alloca_d = ptg.find(test.alloca_d);

		auto & palloca_a = ptg.find_regnode(test.alloca_a->output(0));
		auto & palloca_b = ptg.find_regnode(test.alloca_b->output(0));
		auto & palloca_c = ptg.find_regnode(test.alloca_c->output(0));
		auto & palloca_d = ptg.find_regnode(test.alloca_d->output(0));

		auto & lambda = ptg.find(test.lambda);
		auto & plambda = ptg.find_regnode(test.lambda->output());

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

	store_test1 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);
	validate_ptg(*ptg, test);
}

static void
TestStore2()
{
	auto validate_ptg = [](const jlm::aa::ptg & ptg, const store_test2 & test)
	{
		assert(ptg.nallocnodes() == 6);
		assert(ptg.nregnodes() == 6);

		auto & alloca_a = ptg.find(test.alloca_a);
		auto & alloca_b = ptg.find(test.alloca_b);
		auto & alloca_x = ptg.find(test.alloca_x);
		auto & alloca_y = ptg.find(test.alloca_y);
		auto & alloca_p = ptg.find(test.alloca_p);

		auto & palloca_a = ptg.find_regnode(test.alloca_a->output(0));
		auto & palloca_b = ptg.find_regnode(test.alloca_b->output(0));
		auto & palloca_x = ptg.find_regnode(test.alloca_x->output(0));
		auto & palloca_y = ptg.find_regnode(test.alloca_y->output(0));
		auto & palloca_p = ptg.find_regnode(test.alloca_p->output(0));

		auto & lambda = ptg.find(test.lambda);
		auto & plambda = ptg.find_regnode(test.lambda->output());

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

	store_test2 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);
	validate_ptg(*ptg, test);
}

static void
TestLoad1()
{
	auto validate_ptg = [](const jlm::aa::ptg & ptg, const load_test1 & test)
	{
		assert(ptg.nallocnodes() == 1);
		assert(ptg.nregnodes() == 3);

		auto & pload_p = ptg.find_regnode(test.load_p->output(0));

		auto & lambda = ptg.find(test.lambda);
		auto & plambda = ptg.find_regnode(test.lambda->output());
		auto & lambarg0 = ptg.find_regnode(test.lambda->subregion()->argument(0));

		assertTargets(pload_p, {&ptg.memunknown()});

		assertTargets(plambda, {&lambda});
		assertTargets(lambarg0, {&ptg.memunknown()});
	};

	load_test1 test;
//	jive::view(test.graph()->root(), stdout);
	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);

	validate_ptg(*ptg, test);
}

static void
TestLoad2()
{
	auto validate_ptg = [](const jlm::aa::ptg & ptg, const load_test2 & test)
	{
		assert(ptg.nallocnodes() == 6);
		assert(ptg.nregnodes() == 8);

		/*
			We only care about the loads in this test, skipping the validation
			for all other nodes.
		*/
		auto & alloca_a = ptg.find(test.alloca_a);
		auto & alloca_b = ptg.find(test.alloca_b);
		auto & alloca_x = ptg.find(test.alloca_x);
		auto & alloca_y = ptg.find(test.alloca_y);

		auto & pload_x = ptg.find_regnode(test.load_x->output(0));
		auto & pload_a = ptg.find_regnode(test.load_a->output(0));

		assertTargets(pload_x, {&alloca_x, &alloca_y});
		assertTargets(pload_a, {&alloca_a, &alloca_b});
	};

	load_test2 test;
//	jive::view(test.graph()->root(), stdout);
	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);

	validate_ptg(*ptg, test);
}

static void
TestGetElementPtr()
{
	auto validatePtg = [](const jlm::aa::ptg & ptg, const GetElementPtrTest & test)
	{
		assert(ptg.nallocnodes() == 1);
		assert(ptg.nregnodes() == 4);

		/*
			We only care about the getelemenptr's in this test, skipping the validation
			for all other nodes.
		*/
		auto & gepX = ptg.find_regnode(test.getElementPtrX->output(0));
		auto & gepY = ptg.find_regnode(test.getElementPtrY->output(0));

		assertTargets(gepX, {&ptg.memunknown()});
		assertTargets(gepY, {&ptg.memunknown()});
	};

	GetElementPtrTest test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);
	validatePtg(*ptg, test);
}

static void
TestBitCast()
{
	auto validatePtg = [](const jlm::aa::ptg & ptg, const BitCastTest & test)
	{
		assert(ptg.nallocnodes() == 1);
		assert(ptg.nregnodes() == 3);

		auto & lambda = ptg.find(test.lambda);
		auto & lambdaOut = ptg.find_regnode(test.lambda->output());
		auto & lambdaArg = ptg.find_regnode(test.lambda->fctargument(0));

		auto & bitCast = ptg.find_regnode(test.bitCast->output(0));

		assertTargets(lambdaOut, {&lambda});
		assertTargets(lambdaArg, {&ptg.memunknown()});
		assertTargets(bitCast, {&ptg.memunknown()});
	};

	BitCastTest test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);
	validatePtg(*ptg, test);
}

static void
TestConstantPointerNull()
{
	auto validatePtg = [](const jlm::aa::ptg & ptg, const ConstantPointerNullTest & test)
	{
		assert(ptg.nallocnodes() == 1);
		assert(ptg.nregnodes() == 3);

		auto & lambda = ptg.find(test.lambda);
		auto & lambdaOut = ptg.find_regnode(test.lambda->output());
		auto & lambdaArg = ptg.find_regnode(test.lambda->fctargument(0));

		auto & null = ptg.find_regnode(test.null->output(0));

		assertTargets(lambdaOut, {&lambda});
		assertTargets(lambdaArg, {&ptg.memunknown()});
		assertTargets(null, {&ptg.memunknown()});
	};

	ConstantPointerNullTest test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);
	validatePtg(*ptg, test);
}

static void
TestBits2Ptr()
{
	auto validate_ptg = [](const jlm::aa::ptg & ptg, const bits2ptr_test & test)
	{
		assert(ptg.nallocnodes() == 2);
		assert(ptg.nregnodes() == 5);

		auto & call_out0 = ptg.find_regnode(test.call->output(0));
		assertTargets(call_out0, {&ptg.memunknown()});

		auto & bits2ptr = ptg.find_regnode(test.call->output(0));
		assertTargets(bits2ptr, {&ptg.memunknown()});
	};

	bits2ptr_test test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);
	validate_ptg(*ptg, test);
}

static void
TestCall1()
{
	auto validate_ptg = [](const jlm::aa::ptg & ptg, const call_test1 & test)
	{
		assert(ptg.nallocnodes() == 6);
		assert(ptg.nregnodes() == 12);

		auto & alloca_x = ptg.find(test.alloca_x);
		auto & alloca_y = ptg.find(test.alloca_y);
		auto & alloca_z = ptg.find(test.alloca_z);

		auto & palloca_x = ptg.find_regnode(test.alloca_x->output(0));
		auto & palloca_y = ptg.find_regnode(test.alloca_y->output(0));
		auto & palloca_z = ptg.find_regnode(test.alloca_z->output(0));

		auto & lambda_f = ptg.find(test.lambda_f);
		auto & lambda_g = ptg.find(test.lambda_g);
		auto & lambda_h = ptg.find(test.lambda_h);

		auto & plambda_f = ptg.find_regnode(test.lambda_f->output());
		auto & plambda_g = ptg.find_regnode(test.lambda_g->output());
		auto & plambda_h = ptg.find_regnode(test.lambda_h->output());

		auto & lambda_f_arg0 = ptg.find_regnode(test.lambda_f->subregion()->argument(0));
		auto & lambda_f_arg1 = ptg.find_regnode(test.lambda_f->subregion()->argument(1));

		auto & lambda_g_arg0 = ptg.find_regnode(test.lambda_g->subregion()->argument(0));
		auto & lambda_g_arg1 = ptg.find_regnode(test.lambda_g->subregion()->argument(1));

		auto & lambda_h_cv0 = ptg.find_regnode(test.lambda_h->subregion()->argument(1));
		auto & lambda_h_cv1 = ptg.find_regnode(test.lambda_h->subregion()->argument(2));

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

	call_test1 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);
	validate_ptg(*ptg, test);
}

static void
TestCall2()
{
	auto validate_ptg = [](const jlm::aa::ptg & ptg, const call_test2 & test)
	{
		assert(ptg.nallocnodes() == 4);
		assert(ptg.nimpnodes() == 0);
		assert(ptg.nregnodes() == 11);

		auto & lambda_create = ptg.find(test.lambda_create);
		auto & lambda_create_out = ptg.find_regnode(test.lambda_create->output());

		auto & lambda_destroy = ptg.find(test.lambda_destroy);
		auto & lambda_destroy_out = ptg.find_regnode(test.lambda_destroy->output());
		auto & lambda_destroy_arg = ptg.find_regnode(test.lambda_destroy->fctargument(0));

		auto & lambda_test = ptg.find(test.lambda_test);
		auto & lambda_test_out = ptg.find_regnode(test.lambda_test->output());
		auto & lambda_test_cv1 = ptg.find_regnode(test.lambda_test->cvargument(0));
		auto & lambda_test_cv2 = ptg.find_regnode(test.lambda_test->cvargument(1));

		auto & call_create1_out = ptg.find_regnode(test.call_create1->output(0));
		auto & call_create2_out = ptg.find_regnode(test.call_create2->output(0));

		auto & malloc = ptg.find(test.malloc);
		auto & malloc_out = ptg.find_regnode(test.malloc->output(0));

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

	call_test2 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);
	validate_ptg(*ptg, test);
}

static void
TestIndirectCall()
{
	auto validate_ptg = [](const jlm::aa::ptg & ptg, const indirect_call_test & test)
	{
		assert(ptg.nallocnodes() == 4);
		assert(ptg.nimpnodes() == 0);
		assert(ptg.nregnodes() == 8);

		auto & lambda_three = ptg.find(test.lambda_three);
		auto & lambda_three_out = ptg.find_regnode(test.lambda_three->output());

		auto & lambda_four = ptg.find(test.lambda_four);
		auto & lambda_four_out = ptg.find_regnode(test.lambda_four->output());

		auto & lambda_indcall = ptg.find(test.lambda_indcall);
		auto & lambda_indcall_out = ptg.find_regnode(test.lambda_indcall->output());
		auto & lambda_indcall_arg = ptg.find_regnode(test.lambda_indcall->fctargument(0));

		auto & lambda_test = ptg.find(test.lambda_test);
		auto & lambda_test_out = ptg.find_regnode(test.lambda_test->output());
		auto & lambda_test_cv0 = ptg.find_regnode(test.lambda_test->cvargument(0));
		auto & lambda_test_cv1 = ptg.find_regnode(test.lambda_test->cvargument(1));
		auto & lambda_test_cv2 = ptg.find_regnode(test.lambda_test->cvargument(2));

		assertTargets(lambda_three_out, {&lambda_three, &lambda_four});

		assertTargets(lambda_four_out, {&lambda_three, &lambda_four});

		assertTargets(lambda_indcall_out, {&lambda_indcall});
		assertTargets(lambda_indcall_arg, {&lambda_three, &lambda_four});

		assertTargets(lambda_test_out, {&lambda_test});
		assertTargets(lambda_test_cv0, {&lambda_indcall});
		assertTargets(lambda_test_cv1, {&lambda_three, &lambda_four});
		assertTargets(lambda_test_cv2, {&lambda_three, &lambda_four});
	};

	indirect_call_test test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);
	validate_ptg(*ptg, test);
}

static void
TestGamma()
{
	auto validate_ptg = [](const jlm::aa::ptg & ptg, const gamma_test & test)
	{
		assert(ptg.nallocnodes() == 1);
		assert(ptg.nregnodes() == 15);

		for (size_t n = 1; n < 5; n++) {
			auto & lambda_arg = ptg.find_regnode(test.lambda->fctargument(n));
			assertTargets(lambda_arg, {&ptg.memunknown()});
		}

		for (size_t n = 0; n < 4; n++) {
			auto & argument0 = ptg.find_regnode(test.gamma->entryvar(n)->argument(0));
			auto & argument1 = ptg.find_regnode(test.gamma->entryvar(n)->argument(1));

			assertTargets(argument0, {&ptg.memunknown()});
			assertTargets(argument1, {&ptg.memunknown()});
		}

		for (size_t n = 0; n < 4; n++) {
			auto & output = ptg.find_regnode(test.gamma->exitvar(0));
			assertTargets(output, {&ptg.memunknown()});
		}
	};

	gamma_test test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);
	validate_ptg(*ptg, test);
}

static void
TestTheta()
{
	auto validate_ptg = [](const jlm::aa::ptg & ptg, const theta_test & test)
	{
		assert(ptg.nallocnodes() == 1);
		assert(ptg.nregnodes() == 5);

		auto & lambda = ptg.find(test.lambda);
		auto & lambda_arg1 = ptg.find_regnode(test.lambda->fctargument(1));
		auto & lambda_out = ptg.find_regnode(test.lambda->output());

		auto & gep_out = ptg.find_regnode(test.gep->output(0));

		auto & theta_lv2_arg = ptg.find_regnode(test.theta->output(2)->argument());
		auto & theta_lv2_out = ptg.find_regnode(test.theta->output(2));

		assertTargets(lambda_arg1, {&ptg.memunknown()});
		assertTargets(lambda_out, {&lambda});

		assertTargets(gep_out, {&ptg.memunknown()});

		assertTargets(theta_lv2_arg, {&ptg.memunknown()});
		assertTargets(theta_lv2_out, {&ptg.memunknown()});
	};

	theta_test test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);
	validate_ptg(*ptg, test);
}

static void
TestDelta1()
{
	auto validate_ptg = [](const jlm::aa::ptg & ptg, const delta_test1 & test)
	{
		assert(ptg.nallocnodes() == 3);
		assert(ptg.nregnodes() == 6);

		auto & delta_f = ptg.find(test.delta_f);
		auto & pdelta_f = ptg.find_regnode(test.delta_f->output());

		auto & lambda_g = ptg.find(test.lambda_g);
		auto & plambda_g = ptg.find_regnode(test.lambda_g->output());
		auto & lambda_g_arg0 = ptg.find_regnode(test.lambda_g->fctargument(0));

		auto & lambda_h = ptg.find(test.lambda_h);
		auto & plambda_h = ptg.find_regnode(test.lambda_h->output());
		auto & lambda_h_cv0 = ptg.find_regnode(test.lambda_h->cvargument(0));
		auto & lambda_h_cv1 = ptg.find_regnode(test.lambda_h->cvargument(1));

		assertTargets(pdelta_f, {&delta_f});

		assertTargets(plambda_g, {&lambda_g});
		assertTargets(plambda_h, {&lambda_h});

		assertTargets(lambda_g_arg0, {&delta_f});

		assertTargets(lambda_h_cv0, {&delta_f});
		assertTargets(lambda_h_cv1, {&lambda_g});
	};

	delta_test1 test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);
	validate_ptg(*ptg, test);
}

static void
TestDelta2()
{
	auto validate_ptg = [](const jlm::aa::ptg & ptg, const delta_test2 & test)
	{
		assert(ptg.nallocnodes() == 4);
		assert(ptg.nregnodes() == 8);

		auto & delta_d1 = ptg.find(test.delta_d1);
		auto & delta_d1_out = ptg.find_regnode(test.delta_d1->output());

		auto & delta_d2 = ptg.find(test.delta_d2);
		auto & delta_d2_out = ptg.find_regnode(test.delta_d2->output());

		auto & lambda_f1 = ptg.find(test.lambda_f1);
		auto & lambda_f1_out = ptg.find_regnode(test.lambda_f1->output());
		auto & lambda_f1_cvd1 = ptg.find_regnode(test.lambda_f1->cvargument(0));

		auto & lambda_f2 = ptg.find(test.lambda_f2);
		auto & lambda_f2_out = ptg.find_regnode(test.lambda_f2->output());
		auto & lambda_f2_cvd1 = ptg.find_regnode(test.lambda_f2->cvargument(0));
		auto & lambda_f2_cvd2 = ptg.find_regnode(test.lambda_f2->cvargument(1));
		auto & lambda_f2_cvf1 = ptg.find_regnode(test.lambda_f2->cvargument(2));

		assertTargets(delta_d1_out, {&delta_d1});
		assertTargets(delta_d2_out, {&delta_d2});

		assertTargets(lambda_f1_out, {&lambda_f1});
		assertTargets(lambda_f1_cvd1, {&delta_d1});

		assertTargets(lambda_f2_out, {&lambda_f2});
		assertTargets(lambda_f2_cvd1, {&delta_d1});
		assertTargets(lambda_f2_cvd2, {&delta_d2});
		assertTargets(lambda_f2_cvf1, {&lambda_f1});
	};

	delta_test2 test;
	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
	std::cout << jlm::aa::ptg::to_dot(*ptg);
	validate_ptg(*ptg, test);
}

static void
TestImports()
{
	auto validate_ptg = [](const jlm::aa::ptg & ptg, const import_test & test)
	{
		assert(ptg.nallocnodes() == 2);
		assert(ptg.nimpnodes() == 2);
		assert(ptg.nregnodes() == 8);

		auto & d1 = ptg.find(test.import_d1);
		auto & import_d1 = ptg.find_regnode(test.import_d1);

		auto & d2 = ptg.find(test.import_d2);
		auto & import_d2 = ptg.find_regnode(test.import_d2);

		auto & lambda_f1 = ptg.find(test.lambda_f1);
		auto & lambda_f1_out = ptg.find_regnode(test.lambda_f1->output());
		auto & lambda_f1_cvd1 = ptg.find_regnode(test.lambda_f1->cvargument(0));

		auto & lambda_f2 = ptg.find(test.lambda_f2);
		auto & lambda_f2_out = ptg.find_regnode(test.lambda_f2->output());
		auto & lambda_f2_cvd1 = ptg.find_regnode(test.lambda_f2->cvargument(0));
		auto & lambda_f2_cvd2 = ptg.find_regnode(test.lambda_f2->cvargument(1));
		auto & lambda_f2_cvf1 = ptg.find_regnode(test.lambda_f2->cvargument(2));

		assertTargets(import_d1, {&d1});
		assertTargets(import_d2, {&d2});

		assertTargets(lambda_f1_out, {&lambda_f1});
		assertTargets(lambda_f1_cvd1, {&d1});

		assertTargets(lambda_f2_out, {&lambda_f2});
		assertTargets(lambda_f2_cvd1, {&d1});
		assertTargets(lambda_f2_cvd2, {&d2});
		assertTargets(lambda_f2_cvf1, {&lambda_f1});
	};

	import_test test;
	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
	std::cout << jlm::aa::ptg::to_dot(*ptg);
	validate_ptg(*ptg, test);
}

static void
TestPhi()
{
	auto validate_ptg = [](const jlm::aa::ptg & ptg, const phi_test & test)
	{
		assert(ptg.nallocnodes() == 3);
		assert(ptg.nregnodes() == 16);

		auto & lambda_fib = ptg.find(test.lambda_fib);
		auto & lambda_fib_out = ptg.find_regnode(test.lambda_fib->output());
		auto & lambda_fib_arg1 = ptg.find_regnode(test.lambda_fib->fctargument(1));

		auto & lambda_test = ptg.find(test.lambda_test);
		auto & lambda_test_out = ptg.find_regnode(test.lambda_test->output());

		auto & phi_rv = ptg.find_regnode(test.phi->begin_rv().output());
		auto & phi_rv_arg = ptg.find_regnode(test.phi->begin_rv().output()->argument());

		auto & gamma_result = ptg.find_regnode(test.gamma->subregion(0)->argument(1));
		auto & gamma_fib = ptg.find_regnode(test.gamma->subregion(0)->argument(2));

		auto & alloca = ptg.find(test.alloca);
		auto & alloca_out = ptg.find_regnode(test.alloca->output(0));

		assertTargets(lambda_fib_out, {&lambda_fib});
		assertTargets(lambda_fib_arg1, {&alloca});

		assertTargets(lambda_test_out, {&lambda_test});

		assertTargets(phi_rv, {&lambda_fib});
		assertTargets(phi_rv_arg, {&lambda_fib});

		assertTargets(gamma_result, {&alloca});
		assertTargets(gamma_fib, {&lambda_fib});

		assertTargets(alloca_out, {&alloca});
	};

	phi_test test;
//	jive::view(test.graph().root(), stdout);

	auto ptg = runSteensgaard(test.module());
//	std::cout << jlm::aa::ptg::to_dot(*ptg);
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

JLM_UNIT_TEST_REGISTER("libjlm/opt/alias-analyses/test-steensgaard", test)

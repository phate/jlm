/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"

#include <jive/evaluator/eval.h>
#include <jive/evaluator/literal.h>
#include <jive/types/bitstring/arithmetic.h>
#include <jive/types/bitstring/constant.h>
#include <jive/types/bitstring/comparison.h>
#include <jive/view.h>
#include <jive/rvsdg/gamma.h>
#include <jive/rvsdg/graph.h>
#include <jive/rvsdg/simple-node.h>
#include <jive/rvsdg/theta.h>

#include <jlm/jlm/opt/dne.hpp>
#include <jlm/jlm/opt/unroll.hpp>

static size_t
nthetas(jive::region * region)
{
	size_t n = 0;
	for (const auto & node : region->nodes) {
		if (jive::is<jive::theta_op>(&node))
			n++;
	}

	return n;
}

static jive::theta_node *
create_theta(
	const jive::bitcompare_op & cop,
	const jive::bitbinary_op & aop,
	jive::output * init,
	jive::output * step,
	jive::output * end)
{
	using namespace jive;

	auto graph = init->region()->graph();

	auto theta = theta_node::create(graph->root());
	auto subregion = theta->subregion();
	auto idv = theta->add_loopvar(init);
	auto lvs = theta->add_loopvar(step);
	auto lve = theta->add_loopvar(end);

	auto arm = simple_node::create_normalized(subregion, aop, {idv->argument(), lvs->argument()})[0];
	auto cmp = simple_node::create_normalized(subregion, cop, {arm, lve->argument()})[0];
	auto match = jive::match(1, {{1, 1}}, 0, 2, cmp);

	idv->result()->divert_to(arm);
	theta->set_predicate(match);

	return theta;
}

static inline void
test_unrollinfo()
{
	jive::bittype bt32(32);
	jive::bitslt_op slt(bt32);
	jive::bitult_op ult(bt32);
	jive::bitule_op ule(bt32);
	jive::bitugt_op ugt(bt32);
	jive::bitsge_op sge(bt32);
	jive::biteq_op eq(bt32);

	jive::bitadd_op add(32);
	jive::bitsub_op sub(32);

	{
		jive::graph graph;
		auto x = graph.add_import(bt32, "x");
		auto theta = create_theta(slt, add, x, x, x);
		auto ui = jlm::unrollinfo::create(theta);

		assert(ui);
		assert(ui->is_additive());
		assert(!ui->is_subtractive());
		assert(!ui->is_known());
		assert(!ui->niterations());
		assert(ui->theta() == theta);
		assert(ui->idv()->input()->origin() == x);
	}

	{
		jive::graph graph;
		auto nf = graph.node_normal_form(typeid(jive::operation));
		nf->set_mutable(false);

		auto init0 = jive::create_bitconstant(graph.root(), 32, 0);
		auto init1 = jive::create_bitconstant(graph.root(), 32, 1);
		auto initm1 = jive::create_bitconstant(graph.root(), 32, 0xFFFFFFFF);

		auto step1 = jive::create_bitconstant(graph.root(), 32, 1);
		auto step0 = jive::create_bitconstant(graph.root(), 32, 0);
		auto stepm1 = jive::create_bitconstant(graph.root(), 32, 0xFFFFFFFF);
		auto step2 = jive::create_bitconstant(graph.root(), 32, 2);

		auto end100 = jive::create_bitconstant(graph.root(), 32, 100);

		auto theta = create_theta(ult, add, init0, step1, end100);
		auto ui = jlm::unrollinfo::create(theta);
		assert(ui && *ui->niterations() == 100);

		theta = create_theta(ule, add, init0, step1, end100);
		ui = jlm::unrollinfo::create(theta);
		assert(ui && *ui->niterations() == 101);

		theta = create_theta(ugt, sub, end100, stepm1, init0);
		ui = jlm::unrollinfo::create(theta);
		assert(ui && *ui->niterations() == 100);

		theta = create_theta(sge, sub, end100, stepm1, init0);
		ui = jlm::unrollinfo::create(theta);
		assert(ui && *ui->niterations() == 101);

		theta = create_theta(ult, add, init0, step0, end100);
		ui = jlm::unrollinfo::create(theta);
		assert(ui && !ui->niterations());

		theta = create_theta(eq, add, initm1, step1, end100);
		ui = jlm::unrollinfo::create(theta);
		assert(ui && *ui->niterations() == 101);

		theta = create_theta(eq, add, init1, step2, end100);
		ui = jlm::unrollinfo::create(theta);
		assert(ui && !ui->niterations());
	}
}

static inline void
test_known_boundaries()
{
	jive::bitult_op ult(32);
	jive::bitsgt_op sgt(32);
	jive::bitadd_op add(32);
	jive::bitsub_op sub(32);

	{
		jive::graph graph;
		auto nf = graph.node_normal_form(typeid(jive::operation));
		nf->set_mutable(false);

		auto init = jive::create_bitconstant(graph.root(), 32, 0);
		auto step = jive::create_bitconstant(graph.root(), 32, 1);
		auto end = jive::create_bitconstant(graph.root(), 32, 4);

		auto theta = create_theta(ult, add, init, step, end);
//		jive::view(graph, stdout);
		jlm::unroll(theta, 4);
//		jive::view(graph, stdout);
		/*
			The unroll factor is greater than or equal the number of iterations.
			The loop should be fully unrolled and the theta removed.
		*/
		assert(nthetas(graph.root()) == 0);
	}

	{
		jive::graph graph;
		auto nf = graph.node_normal_form(typeid(jive::operation));
		nf->set_mutable(false);

		auto init = jive::create_bitconstant(graph.root(), 32, 0);
		auto step = jive::create_bitconstant(graph.root(), 32, 1);
		auto end = jive::create_bitconstant(graph.root(), 32, 100);

		auto theta = create_theta(ult, add, init, step, end);
//		jive::view(graph, stdout);
		jlm::unroll(theta, 2);
//		jive::view(graph, stdout);
		/*
			The unroll factor is a multiple of the number of iterations.
			We should only find one (unrolled) theta.
		*/
		assert(nthetas(graph.root()) == 1);
	}

	{
		jive::graph graph;
		auto nf = graph.node_normal_form(typeid(jive::operation));
		nf->set_mutable(false);

		auto init = jive::create_bitconstant(graph.root(), 32, 0);
		auto step = jive::create_bitconstant(graph.root(), 32, 1);
		auto end = jive::create_bitconstant(graph.root(), 32, 100);

		auto theta = create_theta(ult, add, init, step, end);
//		jive::view(graph, stdout);
		jlm::unroll(theta, 3);
//		jive::view(graph, stdout);
		/*
			The unroll factor is NOT a multiple of the number of iterations
			and we have one remaining iteration. We should find only the
			unrolled theta and the body of the old theta as epilogue.
		*/
		assert(nthetas(graph.root()) == 1);
	}

	{
		jive::graph graph;
		auto nf = graph.node_normal_form(typeid(jive::operation));
		nf->set_mutable(false);

		auto init = jive::create_bitconstant(graph.root(), 32, 100);
		auto step = jive::create_bitconstant(graph.root(), 32, -1);
		auto end = jive::create_bitconstant(graph.root(), 32, 0);

		auto theta = create_theta(sgt, sub, init, step, end);
//		jive::view(graph, stdout);
		jlm::unroll(theta, 6);
//		jive::view(graph, stdout);
		/*
			The unroll factor is NOT a multiple of the number of iterations
			and we have four remaining iterations. We should find two thetas:
			one unrolled theta and one theta for the residual iterations.
		*/
		assert(nthetas(graph.root()) == 2);
	}
}

static inline void
test_unknown_boundaries()
{
	jive::bittype bt(32);
	jlm::test_op op({&bt}, {&bt});

	jive::graph graph;
	auto x = graph.add_import(bt, "x");
	auto y = graph.add_import(bt, "y");

	auto theta = jive::theta_node::create(graph.root());
	auto lv1 = theta->add_loopvar(x);
	auto lv2 = theta->add_loopvar(y);

	auto one = jive::create_bitconstant(theta->subregion(), 32, 1);
	auto add = jive::bitadd_op::create(32, lv1->argument(), one);
	auto cmp = jive::bitult_op::create(32, add, lv2->argument());
	auto match = jive::match(1, {{1, 0}}, 1, 2, cmp);

	lv1->result()->divert_to(add);

	theta->set_predicate(match);

	auto ex1 = graph.add_export(lv1, "x");

	jive::view(graph, stdout);
	jlm::unroll(graph, 2);
	jive::view(graph, stdout);

	auto node = ex1->origin()->node();
	assert(jive::is<jive::gamma_op>(node));
	node = node->input(1)->origin()->node();
	assert(jive::is<jive::gamma_op>(node));
}

static int
verify()
{
	test_unrollinfo();

	test_known_boundaries();
	test_unknown_boundaries();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/test-unroll", verify)

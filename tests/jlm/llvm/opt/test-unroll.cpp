/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"

#include <jlm/rvsdg/bitstring/arithmetic.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/bitstring/comparison.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/llvm/opt/unroll.hpp>
#include <jlm/util/Statistics.hpp>

static jlm::StatisticsCollector statisticsCollector;

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
		auto x = graph.add_import({bt32, "x"});
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
	using namespace jlm;

	jive::bittype bt(32);
	jlm::test_op op({&bt}, {&bt});

	RvsdgModule rm(filepath(""), "", "");
	auto & graph = rm.Rvsdg();

	auto x = graph.add_import({bt, "x"});
	auto y = graph.add_import({bt, "y"});

	auto theta = jive::theta_node::create(graph.root());
	auto lv1 = theta->add_loopvar(x);
	auto lv2 = theta->add_loopvar(y);

	auto one = jive::create_bitconstant(theta->subregion(), 32, 1);
	auto add = jive::bitadd_op::create(32, lv1->argument(), one);
	auto cmp = jive::bitult_op::create(32, add, lv2->argument());
	auto match = jive::match(1, {{1, 0}}, 1, 2, cmp);

	lv1->result()->divert_to(add);

	theta->set_predicate(match);

	auto ex1 = graph.add_export(lv1, {lv1->type(), "x"});

//	jive::view(graph, stdout);
	jlm::loopunroll loopunroll(2);
	loopunroll.run(rm, statisticsCollector);
//	jive::view(graph, stdout);

	auto node = jive::node_output::node(ex1->origin());
	assert(jive::is<jive::gamma_op>(node));
	node = jive::node_output::node(node->input(1)->origin());
	assert(jive::is<jive::gamma_op>(node));

	/* Create cleaner output */
	jlm::DeadNodeElimination dne;
	dne.run(rm, statisticsCollector);
//	jive::view(graph, stdout);
}

static std::vector<jive::theta_node*>
find_thetas(jive::region * region)
{
	std::vector<jive::theta_node*> thetas;
	for (auto & node : jive::topdown_traverser(region)) {
		if (auto theta = dynamic_cast<jive::theta_node*>(node))
			thetas.push_back(theta);
	}

	return thetas;
}

static inline void
test_nested_theta()
{
	jlm::RvsdgModule rm(jlm::filepath(""), "", "");
	auto & graph = rm.Rvsdg();

	auto nf = graph.node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto init = jive::create_bitconstant(graph.root(), 32, 0);
	auto step = jive::create_bitconstant(graph.root(), 32, 1);
	auto end = jive::create_bitconstant(graph.root(), 32, 97);

	/* Outer loop */
	auto otheta = jive::theta_node::create(graph.root());

	auto lvo_init = otheta->add_loopvar(init);
	auto lvo_step = otheta->add_loopvar(step);
	auto lvo_end = otheta->add_loopvar(end);

	auto add = jive::bitadd_op::create(32, lvo_init->argument(), lvo_step->argument());
	auto compare = jive::bitult_op::create(32, add, lvo_end->argument());
	auto match = jive::match(1, {{1, 1}}, 0, 2, compare);
	otheta->set_predicate(match);
	lvo_init->result()->divert_to(add);

	/* First inner loop in the original loop */
	auto inner_theta = jive::theta_node::create(otheta->subregion());

	auto inner_init = jive::create_bitconstant(otheta->subregion(), 32, 0);
	auto lvi_init = inner_theta->add_loopvar(inner_init);
	auto lvi_step = inner_theta->add_loopvar(lvo_step->argument());
	auto lvi_end = inner_theta->add_loopvar(lvo_end->argument());

	auto inner_add = jive::bitadd_op::create(32, lvi_init->argument(), lvi_step->argument());
	auto inner_compare = jive::bitult_op::create(32, inner_add, lvi_end->argument());
	auto inner_match = jive::match(1, {{1, 1}}, 0, 2, inner_compare);
	inner_theta->set_predicate(inner_match);
	lvi_init->result()->divert_to(inner_add);

	/* Nested inner loop */
	auto inner_nested_theta = jive::theta_node::create(inner_theta->subregion());

	auto inner_nested_init = jive::create_bitconstant(inner_theta->subregion(), 32, 0);
	auto lvi_nested_init = inner_nested_theta->add_loopvar(inner_nested_init);
	auto lvi_nested_step = inner_nested_theta->add_loopvar(lvi_step->argument());
	auto lvi_nested_end = inner_nested_theta->add_loopvar(lvi_end->argument());

	auto inner_nested_add = jive::bitadd_op::create(32, lvi_nested_init->argument(), lvi_nested_step->argument());
	auto inner_nested_compare = jive::bitult_op::create(32, inner_nested_add, lvi_nested_end->argument());
	auto inner_nested_match = jive::match(1, {{1, 1}}, 0, 2, inner_nested_compare);
	inner_nested_theta->set_predicate(inner_nested_match);
	lvi_nested_init->result()->divert_to(inner_nested_add);
	
	/* Second inner loop in the original loop */
	auto inner2_theta = jive::theta_node::create(otheta->subregion());

	auto inner2_init = jive::create_bitconstant(otheta->subregion(), 32, 0);
	auto lvi2_init = inner2_theta->add_loopvar(inner2_init);
	auto lvi2_step = inner2_theta->add_loopvar(lvo_step->argument());
	auto lvi2_end = inner2_theta->add_loopvar(lvo_end->argument());

	auto inner2_add = jive::bitadd_op::create(32, lvi2_init->argument(), lvi2_step->argument());
	auto inner2_compare = jive::bitult_op::create(32, inner2_add, lvi2_end->argument());
	auto inner2_match = jive::match(1, {{1, 1}}, 0, 2, inner2_compare);
	inner2_theta->set_predicate(inner2_match);
	lvi2_init->result()->divert_to(inner2_add);

	
//	jive::view(graph, stdout);
	jlm::loopunroll loopunroll(4);
	loopunroll.run(rm, statisticsCollector);
	/*
		The outher theta should contain two inner thetas
	*/
	assert(nthetas(otheta->subregion()) == 2);
	/*
		The outer theta should not be unrolled and since the
		original graph contains 7 nodes and the unroll factor
		is 4 an unrolled theta should have around 28 nodes. So
		we check for less than 20 nodes in case an updated
		unroll algorithm would hoist code from the innner
		thetas.
	*/
	assert(otheta->subregion()->nnodes() <= 20);
	/*
		The inner theta should not be unrolled and since the
		original graph contains 5 nodes and the unroll factor
		is 4 an unrolled theta should have around 20 nodes. So
		we check for less than 15 nodes in case an updated
		unroll algorithm would hoist code from the innner
		thetas.
	*/
	assert(inner_theta->subregion()->nnodes() <= 15);
	/*
		The innermost theta should be unrolled and since the
		original graph contains 3 nodes and the unroll factor
		is 4 an unrolled theta should have around 12 nodes. So
		we check for more than 7 nodes in case an updated
		unroll algorithm would hoist code from the innner
		thetas.
	*/
	auto thetas = find_thetas(inner_theta->subregion());
	assert(thetas.size() == 1 && thetas[0]->subregion()->nnodes() >= 7);
	/*
		The second inner theta should be unrolled and since
		the original graph contains 3 nodes and the unroll
		factor is 4 an unrolled theta should have around 12
		nodes. So we check for less than 7 nodes in case an
		updated unroll algorithm would hoist code from the
		innner thetas.
	*/
	thetas = find_thetas(otheta->subregion());
	assert(thetas.size() == 2 && thetas[1]->subregion()->nnodes() >= 7);
//	jive::view(graph, stdout);
	jlm::unroll(otheta, 4);
//	jive::view(graph, stdout);
	/*
		After unrolling the outher theta four times it should
		now contain 8 thetas.
	*/
	thetas = find_thetas(graph.root());
	assert(thetas.size() == 3 && nthetas(thetas[0]->subregion()) == 8);
}


static int
verify()
{
	test_unrollinfo();

	test_nested_theta();
	test_known_boundaries();
	test_unknown_boundaries();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-unroll", verify)

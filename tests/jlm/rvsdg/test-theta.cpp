/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

static int
test_main()
{
	using namespace jive;

	jive::graph graph;
	jlm::valuetype t;

	auto imp1 = graph.add_import({ctl2, "imp1"});
	auto imp2 = graph.add_import({t, "imp2"});
	auto imp3 = graph.add_import({t, "imp3"});

	auto theta = jive::theta_node::create(graph.root());

	auto lv1 = theta->add_loopvar(imp1);
	auto lv2 = theta->add_loopvar(imp2);
	auto lv3 = theta->add_loopvar(imp3);

	lv2->result()->divert_to(lv3->argument());
	lv3->result()->divert_to(lv3->argument());
	theta->set_predicate(lv1->argument());

	graph.add_export(theta->output(0), {theta->output(0)->type(), "exp"});
	auto theta2 = static_cast<jive::structural_node*>(theta)->copy(graph.root(), {imp1, imp2, imp3});
	jive::view(graph.root(), stdout);

	assert(lv1->node() == theta);
	assert(lv2->node() == theta);
	assert(lv3->node() == theta);

	assert(theta->predicate() == theta->subregion()->result(0));
	assert(theta->nloopvars() == 3);
	assert((*theta->begin())->result() == theta->subregion()->result(1));

	assert(dynamic_cast<const jive::theta_node*>(theta2));

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-theta", test_main)

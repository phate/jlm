/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-types.hpp"

#include <jive/view.h>
#include <jive/vsdg/controltype.h>
#include <jive/vsdg/gamma.h>
#include <jive/vsdg/graph.h>
#include <jive/vsdg/theta.h>

#include <jlm/opt/invariance.hpp>

static inline void
test_gamma()
{
	jlm::valuetype vt;
	jive::ctl::type ct(2);

	jive::graph graph;
	auto c = graph.import(ct, "c");
	auto x = graph.import(vt, "x");
	auto y = graph.import(vt, "y");

	jive::gamma_builder gb1;
	gb1.begin(c);
	auto ev1 = gb1.add_entryvar(c);
	auto ev2 = gb1.add_entryvar(x);
	auto ev3 = gb1.add_entryvar(y);

	jive::gamma_builder gb2;
	gb2.begin(ev1->argument(0));
	auto ev4 = gb2.add_entryvar(ev2->argument(0));
	auto ev5 = gb2.add_entryvar(ev3->argument(0));
	auto xv4 = gb2.add_exitvar({ev4->argument(0), ev4->argument(1)});
	auto xv5 = gb2.add_exitvar({ev5->argument(0), ev5->argument(1)});
	auto gamma2 = gb2.end();

	auto xv2 = gb1.add_exitvar({gamma2->node()->output(0), ev2->argument(1)});
	auto xv3 = gb1.add_exitvar({gamma2->node()->output(1), ev3->argument(1)});
	auto gamma1 = gb1.end();

	graph.export_port(gamma1->node()->output(0), "x");
	graph.export_port(gamma1->node()->output(1), "y");

	jive::view(graph.root(), stdout);
	jlm::invariance(graph);
	jive::view(graph.root(), stdout);

	assert(graph.root()->result(0)->origin() == graph.root()->argument(1));
	assert(graph.root()->result(1)->origin() == graph.root()->argument(2));
}

static inline void
test_theta()
{
	jlm::valuetype vt;
	jive::ctl::type ct(2);

	jive::graph graph;
	auto c = graph.import(ct, "c");
	auto x = graph.import(vt, "x");

	jive::theta_builder tb1;
	tb1.begin(graph.root());
	auto lv1 = tb1.add_loopvar(c);
	auto lv2 = tb1.add_loopvar(x);

	jive::theta_builder tb2;
	tb2.begin(tb1.subregion());
	auto lv3 = tb2.add_loopvar(lv1->argument());
	auto lv4 = tb2.add_loopvar(lv2->argument());
	tb2.end(lv3->argument());

	tb1.end(lv1->argument());

	graph.export_port(lv1->output(), "c");
	graph.export_port(lv2->output(), "x");

	jive::view(graph.root(), stdout);
	jlm::invariance(graph);
	jive::view(graph.root(), stdout);

	assert(graph.root()->result(0)->origin() == lv1->output());
	assert(graph.root()->result(1)->origin() == graph.root()->argument(1));
}

static int
verify()
{
	test_gamma();
	test_theta();
	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-invariance", verify);

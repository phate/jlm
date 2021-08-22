/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/ir/operators.hpp>

#include <jive/rvsdg/theta.hpp>
#include <jive/view.hpp>

static void
test_trace_function_input1()
{
	using namespace jlm;

	valuetype vt;
	jive::fcttype fcttype1({}, {&vt});
	ptrtype pt(fcttype1);
	jive::fcttype fcttype2({&pt}, {&vt});

	auto module = rvsdg_module::create(filepath(""), "", "");
	auto graph = module->graph();

	auto nf = graph->node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	auto fct = lambda::node::create(graph->root(), fcttype2, "fct", linkage::external_linkage);

	auto one = jive::create_bitconstant(fct->subregion(), 32, 1);

	auto alloca = alloca_op::create(pt, one, 8);

	auto store = store_op::create(alloca[0], fct->fctargument(0), {alloca[1]}, 8);

	auto load = load_op::create(alloca[0], store, 8);

	auto call = call_op::create(load[0], {});

	fct->finalize({call[0]});

	graph->add_export(fct->output(), {ptrtype(fct->type()), "f"});

	auto callnode = static_cast<const jive::simple_node*>(jive::node_output::node(call[0]));
	assert(load[0] == trace_function_input(*callnode));
}

static void
test_trace_function_input2()
{
	using namespace jlm;

	// Arrange
	valuetype vt;
	jive::fcttype gtype({}, {&vt});
	jive::fcttype ftype({}, {&vt});

	auto module = rvsdg_module::create(filepath(""), "", "");
	auto graph = module->graph();

	auto nf = graph->node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	/* function g */
	auto g = lambda::node::create(graph->root(), gtype, "g", linkage::external_linkage);
	auto constant = test_op::create(g->subregion(), {}, {&vt});
	g->finalize({constant->output(0)});

	/* function f */
	auto f = lambda::node::create(graph->root(), ftype, "f", linkage::external_linkage);
	auto ctxg = f->add_ctxvar(g->output());

	auto outerTheta = jive::theta_node::create(f->subregion());
	auto otf = outerTheta->add_loopvar(ctxg);

	auto innerTheta = jive::theta_node::create(outerTheta->subregion());
	auto itf = innerTheta->add_loopvar(otf->argument());

	auto predicate = jive_control_false(innerTheta->subregion());
	auto gamma = jive::gamma_node::create(predicate, 2);
	auto ev = gamma->add_entryvar(itf->argument());
	auto xv = gamma->add_exitvar({ev->argument(0), ev->argument(1)});

	itf->result()->divert_to(xv);
	otf->result()->divert_to(itf);

	auto call = call_op::create(otf, {});

	f->finalize(call);
	graph->add_export(f->output(), {ptrtype(f->type()), "f"});

	jive::view(graph->root(), stdout);

	// Act
	auto callNode = jive::node_output::node(call[0]);
	auto tracedOutput = trace_function_input(*static_cast<const jive::simple_node*>(callNode));

	// Assert
	assert(tracedOutput == g->output());
}

static int
test()
{
	test_trace_function_input1();
	test_trace_function_input2();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/ir/operators/test-call", test)

/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/ir/operators.hpp>

static void
test_trace_function_input()
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

static int
test()
{
	test_trace_function_input();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/ir/operators/test-call", test)

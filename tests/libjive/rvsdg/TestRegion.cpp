/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-operation.hpp"
#include "test-types.hpp"

#include <assert.h>

#include <jive/rvsdg.hpp>

static int
test_main()
{
	using namespace jive;

	jlm::valuetype vt;

	jive::graph graph;
	auto import1 = graph.add_import({vt, "import1"});

	auto n1 = jlm::structural_node::create(graph.root(), 1);
	auto n2 = jlm::structural_node::create(graph.root(), 2);

	/* Test type error check for adding argument to wrong input */

	auto structi1 = structural_input::create(n1, import1, vt);

	bool input_error_handler_called = false;
	try {
		argument::create(n2->subregion(0), structi1, vt);
	} catch (jive::compiler_error & e) {
		input_error_handler_called = true;
	}

	assert(input_error_handler_called);

	/* Test type error check for adding result to wrong output */

	auto argument = argument::create(n1->subregion(0), structi1, vt);
	auto structo1 = structural_output::create(n1, vt);

	bool output_error_handler_called = false;
	try {
		result::create(n2->subregion(0), argument, structo1, vt);
	} catch (jive::compiler_error & e) {
		output_error_handler_called = true;
	}

	assert(output_error_handler_called);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjive/rvsdg/TestRegion", test_main)

/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-operation.hpp"
#include "test-types.hpp"

#include <assert.h>

static int test_main(void)
{
	using namespace jive;

	jive::graph graph;
	
	jlm::statetype type;
	jlm::valuetype value_type;

	auto n1 = jlm::test_op::create(graph.root(), {}, {&type});

	bool error_handler_called = false;
	try {
		jlm::test_op::Create(graph.root(), {&value_type}, {n1->output(0)}, {});
	} catch (jive::type_error & e) {
		error_handler_called = true;
	}
	
	assert(error_handler_called);

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-typemismatch", test_main)

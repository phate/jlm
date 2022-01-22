/*
 * Copyright 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-operation.hpp"
#include "test-types.hpp"

#include <assert.h>

#include <jive/rvsdg.hpp>
#include <jive/rvsdg/phi.hpp>
#include <jive/types/bitstring.hpp>
#include <jive/types/function.hpp>
#include <jive/view.hpp>

static int
function_test_build_lambda(void)
{
	using namespace jive;

	jive::graph graph;
	auto x = graph.add_import({bit32, "x"});

	jive::lambda_builder lb;
	auto arguments = lb.begin_lambda(graph.root(), {{&bit32, &bit32}, {&bit32}});
	lb.add_dependency(x);

	auto sum = bitadd_op::create(32, arguments[0], arguments[1]);

	auto f1 = lb.end_lambda({sum});
	auto f2 = static_cast<jive::structural_node*>(f1)->copy(graph.root(), {x});

	jive::view(graph.root(), stderr);
	
	assert(f1->output(0)->type() == fcttype({&bit32, &bit32}, {&bit32}));
	assert(dynamic_cast<const jive::lambda_node*>(f2));
	
	return 0;
}

static int function_test_call(void)
{
	using namespace jive;

	jive::graph graph;

	fcttype ftype({&bit8}, {&bit8}) ;

	auto constant = create_bitconstant(graph.root(), "00001111");
	auto func = graph.add_import({ftype, "sin"});
	auto ret = create_apply(func, {constant})[0];

	assert(ret->type() == bit8);

	jive::view(graph.root(), stderr) ;

	return 0 ;
}

static int function_test_equals(void)
{
	using namespace jive;

	fcttype type0({&bit8}, {&bit8});
	fcttype type1({&bit8}, {&bit8});
	fcttype type2({&bit8}, {&bit8, &bit8});
	fcttype type3({&bit8, &bit8}, {&bit8});

	assert(type0 == type0);
	assert(type0 == type1);
	assert(type0 != type2);
	assert(type0 != type3);
	
	return 0 ;
}

static int function_test_memory_leak(void)
{
	jlm::valuetype value_type;
	jive::fcttype t1({&value_type}, {&value_type});
	jive::fcttype t2({&t1}, {&t1});
	
	return 0;
}

static int
RunTests()
{
	function_test_equals();
	function_test_call();
	function_test_build_lambda();
	function_test_memory_leak();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjive/types/function", RunTests)

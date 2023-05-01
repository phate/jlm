/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/ir/operators/operators.hpp>

static void
test_equality()
{
	using namespace jlm;

	ConstantFP c1(fpsize::half, llvm::APFloat(0.0));
	ConstantFP c2(fpsize::flt, llvm::APFloat(0.0));
	ConstantFP c3(fpsize::flt, llvm::APFloat(-0.0));

	assert(c1 != c2);
	assert(c2 != c3);
}

static int
test()
{
	test_equality();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/test-ConstantFP", test)

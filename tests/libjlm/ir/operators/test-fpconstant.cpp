/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/ir/operators/operators.hpp>

static void
test_equality()
{
	using namespace jlm;

	fpconstant_op fp1(fpsize::half, llvm::APFloat(0.0));
	fpconstant_op fp2(fpsize::flt, llvm::APFloat(0.0));

	assert(fp1 != fp2);
}

static int
test()
{
	test_equality();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/ir/operators/test-fpconstant", test);

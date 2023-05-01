/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <jlm/llvm/backend/jlm2llvm/context.hpp>
#include <jlm/llvm/backend/jlm2llvm/type.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>

static void
test_structtype(jlm::jlm2llvm::context & ctx)
{
	using namespace jlm;

	auto decl1 = jive::rcddeclaration::create({&jive::bit8, &jive::bit32});
	StructType st1("mystruct", false, *decl1);
	auto ct = jlm2llvm::convert_type(st1, ctx);

	assert(ct->getName() == "mystruct");
	assert(!ct->isPacked());
	assert(ct->getNumElements() == 2);
	assert(ct->getElementType(0)->isIntegerTy(8));
	assert(ct->getElementType(1)->isIntegerTy(32));

	auto decl2 = jive::rcddeclaration::create({&jive::bit32, &jive::bit8, &jive::bit32});
	StructType st2(true, *decl2);
	ct = jlm2llvm::convert_type(st2, ctx);

	assert(ct->getName().empty());
	assert(ct->isPacked());
	assert(ct->getNumElements() == 3);
}

static int
test()
{
	using namespace jlm;

	llvm::LLVMContext ctx;
	llvm::Module lm("module", ctx);

	ipgraph_module im(filepath(""), "", "");
	jlm2llvm::context jctx(im, lm);

	test_structtype(jctx);

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/backend/llvm/jlm-llvm/test-type-conversion", test)

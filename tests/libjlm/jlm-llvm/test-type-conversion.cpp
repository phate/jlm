/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/types.hpp>
#include <jlm/jlm2llvm/context.hpp>
#include <jlm/jlm2llvm/type.hpp>

static void
test_structtype(jlm::jlm2llvm::context & ctx)
{
	using namespace jlm;

	auto decl1 = jive::rcddeclaration::create({&jive::bit8, &jive::bit32});
	structtype st1("mystruct", false, decl1.get());
	auto ct = jlm2llvm::convert_type(st1, ctx);

	assert(ct->getName() == "mystruct");
	assert(!ct->isPacked());
	assert(ct->getNumElements() == 2);
	assert(ct->getElementType(0)->isIntegerTy(8));
	assert(ct->getElementType(1)->isIntegerTy(32));

	auto decl2 = jive::rcddeclaration::create({&jive::bit32, &jive::bit8, &jive::bit32});
	structtype st2(true, decl2.get());
	ct = jlm2llvm::convert_type(st2, ctx);

	assert(ct->getName() == "");
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

JLM_UNIT_TEST_REGISTER("libjlm/jlm-llvm/test-type-conversion", test)

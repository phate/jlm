/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-operation.hpp"
#include "test-types.hpp"

#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/print.hpp>
#include <jlm/jlm2llvm/jlm2llvm.hpp>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

static int
test()
{
	using namespace jlm;

	valuetype vt;
	ptrtype pt(vt);
	jive::memtype mt;
	ipgraph_module m(filepath(""), "", "");

	std::unique_ptr<jlm::cfg> cfg(new jlm::cfg(m));
	auto bb = basic_block::create(*cfg);
	cfg->exit()->divert_inedges(bb);
	bb->add_outedge(cfg->exit());

	auto p = cfg->entry()->append_argument(argument::create("p", jive::bit1));
	auto s1 = cfg->entry()->append_argument(argument::create("s1", mt));
	auto s2 = cfg->entry()->append_argument(argument::create("s2", mt));

	auto s3 = m.create_variable(mt, "s3");

	bb->append_last(select_op::create(p, s1, s2, s3));

	cfg->exit()->append_result(s3);
	cfg->exit()->append_result(s3);

	jive::fcttype ft({&jive::bit1, &mt, &mt}, {&mt, &mt});
	auto f = function_node::create(m.ipgraph(), "f", ft, linkage::external_linkage);
	f->add_cfg(std::move(cfg));

	print(m, stdout);

	llvm::LLVMContext ctx;
	jlm2llvm::convert(m, ctx);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/jlm-llvm/test-select-with-state", test)

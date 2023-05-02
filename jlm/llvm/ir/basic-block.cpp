/*
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>

#include <sstream>

namespace jlm {

/* basic block */

basic_block::~basic_block()
{}

jlm::tac *
basic_block::insert_before_branch(std::unique_ptr<jlm::tac> tac)
{
	auto it = is<branch_op>(last()) ? std::prev(end()) : end();
	return insert_before(it, std::move(tac));
}

void
basic_block::insert_before_branch(tacsvector_t & tv)
{
	auto it = is<branch_op>(last()) ? std::prev(end()) : end();
	insert_before(it, tv);
}

basic_block *
basic_block::create(jlm::cfg & cfg)
{
	std::unique_ptr<basic_block> node(new basic_block(cfg));
	return static_cast<basic_block*>(cfg.add_node(std::move(node)));
}

}

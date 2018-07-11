/*
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/jlm/ir/basic-block.hpp>
#include <jlm/jlm/ir/cfg.hpp>
#include <jlm/jlm/ir/operators/operators.hpp>
#include <jlm/jlm/ir/tac.hpp>
#include <jlm/jlm/ir/variable.hpp>

#include <sstream>

namespace jlm {

/* taclist */

taclist::~taclist()
{
	for (const auto & tac : tacs_)
		delete tac;
}

/* basic block */

basic_block::~basic_block()
{}

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

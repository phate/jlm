/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/jlm/ir/basic-block.hpp>
#include <jlm/jlm/ir/cfg.hpp>
#include <jlm/jlm/ir/cfg-structure.hpp>
#include <jlm/jlm/ir/cfg-node.hpp>
#include <jlm/jlm/ir/ipgraph.hpp>
#include <jlm/jlm/ir/operators/operators.hpp>
#include <jlm/jlm/ir/tac.hpp>

#include <algorithm>
#include <deque>
#include <sstream>
#include <unordered_map>

#include <stdio.h>
#include <stdlib.h>
#include <sstream>

namespace jlm {

/* cfg entry node */

entry_node::~entry_node()
{}

/* cfg exit node */

exit_node::~exit_node()
{}

/* cfg */

cfg::cfg(jlm::module & module)
: module_(module)
{
	entry_ = std::unique_ptr<entry_node>(new entry_node(*this));
	exit_ = std::unique_ptr<exit_node>(new exit_node(*this));
	entry_->add_outedge(exit_.get());
}

cfg::iterator
cfg::remove_node(cfg::iterator & it)
{
	if (&it->cfg() != this)
		throw jlm::error("node does not belong to this CFG.");

	if (it->ninedges())
		throw jlm::error("cannot remove node. It has still incoming edges.");

	it->remove_outedges();
	std::unique_ptr<basic_block> tmp(it.node());
	auto rit = iterator(std::next(nodes_.find(tmp)));
	nodes_.erase(tmp);
	tmp.release();
	return rit;
}

}

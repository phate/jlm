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

/* entry attribute */

static inline jlm::cfg_node *
create_entry_node(jlm::cfg * cfg)
{
	return cfg_node::create(*cfg, std::make_unique<jlm::entry>());
}

entry::~entry()
{}

/* exit attribute */

static inline jlm::cfg_node *
create_exit_node(jlm::cfg * cfg)
{
	return cfg_node::create(*cfg, std::make_unique<jlm::exit>());
}

exit::~exit()
{}

/* cfg */

cfg::cfg(jlm::module & module)
: module_(module)
{
	entry_ = create_entry_node(this);
	exit_ = create_exit_node(this);
	entry_->add_outedge(exit_);
}

cfg::iterator
cfg::remove_node(cfg::iterator & it)
{
	if (it->cfg() != this)
		throw jlm::error("node does not belong to this CFG.");

	if (it->ninedges())
		throw jlm::error("cannot remove node. It has still incoming edges.");

	it->remove_outedges();
	std::unique_ptr<jlm::cfg_node> tmp(it.node());
	auto rit = iterator(std::next(nodes_.find(tmp)));
	nodes_.erase(tmp);
	tmp.release();
	return rit;
}

}

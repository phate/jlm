/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/tooling/passgraph.hpp>

#include <deque>
#include <functional>

namespace jlm {

/* entry command */

class entry_cmd final : public Command {
public:
	virtual std::string
	ToString() const override
	{
		return "ENTRY";
	}

	virtual void
	Run() const override
	{}
};

/* exit command */

class exit_cmd final : public Command {
public:
	virtual std::string
	ToString() const override
	{
		return "EXIT";
	}

	virtual void
	Run() const override
	{}
};

/* passgraph node */

passgraph_node *
passgraph_node::create(
	passgraph * pgraph,
	std::unique_ptr<Command> cmd)
{
	std::unique_ptr<passgraph_node> pgnode(new passgraph_node(pgraph, std::move(cmd)));
	auto ptr = pgnode.get();
	pgraph->add_node(std::move(pgnode));
	return ptr;
}

/* passgraph */

passgraph::passgraph()
{
	entry_ = passgraph_node::create(this, std::make_unique<entry_cmd>());
	exit_ = passgraph_node::create(this, std::make_unique<exit_cmd>());
}

void
passgraph::run() const
{
	for (const auto & node : topsort(this))
    node->cmd().Run();
}

/* support methods */

std::vector<passgraph_node*>
topsort(const passgraph * pgraph)
{
	std::vector<passgraph_node*> nodes({pgraph->entry()});
	std::deque<passgraph_node*> to_visit({pgraph->entry()});
	std::unordered_set<passgraph_node*> visited({pgraph->entry()});

	while (!to_visit.empty()) {
		auto node = to_visit.front();
		to_visit.pop_front();

		for (const auto & edge : *node) {
			if (visited.find(edge.sink()) == visited.end()) {
				to_visit.push_back(edge.sink());
				visited.insert(edge.sink());
				nodes.push_back(edge.sink());
			}
		}
	}

	return nodes;
}

}

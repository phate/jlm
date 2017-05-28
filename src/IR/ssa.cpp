/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/cfg.hpp>
#include <jlm/IR/cfg_node.hpp>
#include <jlm/IR/operators.hpp>
#include <jlm/IR/ssa.hpp>
#include <jlm/IR/tac.hpp>

#include <unordered_set>

namespace jlm {

void
destruct_ssa(jlm::cfg & cfg)
{
	/* find all blocks containing phis */
	std::unordered_set<cfg_node*> phi_blocks;
	for (auto & node : cfg) {
		if (!is_basic_block(&node))
			continue;

		auto attr = static_cast<basic_block*>(&node.attribute());
		if (attr->ntacs() != 0 && dynamic_cast<const phi_op*>(&attr->first()->operation()))
			phi_blocks.insert(&node);
	}

	/* eliminate phis */
	for (auto phi_block : phi_blocks) {
		auto ass_block = create_basic_block_node(&cfg);
		auto ass_attr = static_cast<basic_block*>(&ass_block->attribute());
		auto phi_attr = static_cast<basic_block*>(&phi_block->attribute());

		while (phi_attr->first()) {
			auto tac = phi_attr->first();
			if (!dynamic_cast<const phi_op*>(&tac->operation()))
				break;

			const phi_op * phi = static_cast<const phi_op*>(&tac->operation());
			const variable * v = cfg.create_variable(phi->type());

			size_t n = 0;
			const variable * value = nullptr;
			std::list<cfg_edge*> edges = phi_block->inedges();
			for (auto it = edges.begin(); it != edges.end(); it++, n++) {
				auto edge_block = (*it)->split();
				auto edge_attr = static_cast<basic_block*>(&edge_block->attribute());

				auto ass = create_assignment(v->type(), tac->input(n), v);
				value = edge_attr->append(std::move(ass))->output(0);
			}
			ass_attr->append(create_assignment(tac->output(0)->type(), value, tac->output(0)));
			phi_attr->drop_first();
		}

		phi_block->divert_inedges(ass_block);
		ass_block->add_outedge(phi_block, 0);
	}
}

}

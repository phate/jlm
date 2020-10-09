/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/basic-block.hpp>
#include <jlm/ir/cfg.hpp>
#include <jlm/ir/cfg-structure.hpp>
#include <jlm/ir/cfg-node.hpp>
#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/operators/operators.hpp>
#include <jlm/ir/ssa.hpp>
#include <jlm/ir/tac.hpp>

#include <unordered_set>

namespace jlm {

void
destruct_ssa(jlm::cfg & cfg)
{
	JLM_DEBUG_ASSERT(is_valid(cfg));

	/* find all blocks containing phis */
	std::unordered_set<cfg_node*> phi_blocks;
	for (auto & node : cfg) {
		if (!is<basic_block>(&node))
			continue;

		auto & tacs = static_cast<basic_block*>(&node)->tacs();
		if (tacs.ntacs() != 0 && dynamic_cast<const phi_op*>(&tacs.first()->operation()))
			phi_blocks.insert(&node);
	}

	/* eliminate phis */
	for (auto phi_block : phi_blocks) {
		auto ass_block = basic_block::create(cfg);
		auto & tacs = static_cast<basic_block*>(phi_block)->tacs();

		/* collect inedges of phi block */
		std::unordered_map<cfg_node*, cfg_edge*> edges;
		for (auto it = phi_block->begin_inedges(); it != phi_block->end_inedges(); it++) {
			JLM_DEBUG_ASSERT(edges.find((*it)->source()) == edges.end());
			edges[(*it)->source()] = *it;
		}

		while (tacs.first()) {
			auto tac = tacs.first();
			if (!dynamic_cast<const phi_op*>(&tac->operation()))
				break;

			auto phi = static_cast<const phi_op*>(&tac->operation());
			auto v = cfg.module().create_variable(phi->type());

			const variable * value = nullptr;
			for (size_t n = 0; n < tac->noperands(); n++) {
				JLM_DEBUG_ASSERT(edges.find(phi->node(n)) != edges.end());
				auto bb = edges[phi->node(n)]->split();
				value = bb->append_last(assignment_op::create(tac->operand(n), v))->operand(0);
			}

			ass_block->append_last(assignment_op::create(value, tac->result(0)));
			tacs.drop_first();
		}

		phi_block->divert_inedges(ass_block);
		ass_block->add_outedge(phi_block);
	}
}

}

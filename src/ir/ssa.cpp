/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/basic_block.hpp>
#include <jlm/ir/cfg.hpp>
#include <jlm/ir/cfg-structure.hpp>
#include <jlm/ir/cfg_node.hpp>
#include <jlm/ir/module.hpp>
#include <jlm/ir/operators.hpp>
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
		if (!is_basic_block(&node))
			continue;

		auto attr = static_cast<basic_block*>(&node.attribute());
		if (attr->ntacs() != 0 && dynamic_cast<const phi_op*>(&attr->first()->operation()))
			phi_blocks.insert(&node);
	}

	/* eliminate phis */
	for (auto phi_block : phi_blocks) {
		auto ass_block = create_basic_block_node(&cfg);
		auto phi_attr = static_cast<basic_block*>(&phi_block->attribute());

		while (phi_attr->first()) {
			auto tac = phi_attr->first();
			if (!dynamic_cast<const phi_op*>(&tac->operation()))
				break;

			auto phi = static_cast<const phi_op*>(&tac->operation());
			auto v = cfg.module().create_variable(phi->type(), false);

			std::unordered_map<cfg_node*, cfg_edge*> edges;
			for (auto it = phi_block->begin_inedges(); it != phi_block->end_inedges(); it++) {
				JLM_DEBUG_ASSERT(edges.find((*it)->source()) == edges.end());
				edges[(*it)->source()] = *it;
			}

			const variable * value;
			for (size_t n = 0; n < tac->ninputs(); n++) {
				JLM_DEBUG_ASSERT(edges.find(phi->node(n)) != edges.end());
				auto bb = edges[phi->node(n)]->split();
				value = append_last(bb, create_assignment(v->type(), tac->input(n), v))->output(0);
			}

			append_last(ass_block, create_assignment(tac->output(0)->type(), value, tac->output(0)));
			phi_attr->drop_first();
		}

		phi_block->divert_inedges(ass_block);
		ass_block->add_outedge(phi_block);
	}
}

}

/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/ssa.hpp>

#include <unordered_set>

namespace jlm::llvm
{

void
destruct_ssa(ControlFlowGraph & cfg)
{
  JLM_ASSERT(is_valid(cfg));

  auto collect_phi_blocks = [](ControlFlowGraph & cfg)
  {
    std::unordered_set<BasicBlock *> phi_blocks;
    for (auto & bb : cfg)
    {
      if (is<SsaPhiOperation>(bb.first()))
        phi_blocks.insert(&bb);
    }

    return phi_blocks;
  };

  auto eliminate_phis =
      [](ControlFlowGraph & cfg, const std::unordered_set<BasicBlock *> & phi_blocks)
  {
    if (phi_blocks.empty())
      return;

    auto firstbb = static_cast<BasicBlock *>(cfg.entry()->OutEdge(0)->sink());

    for (auto phi_block : phi_blocks)
    {
      auto ass_block = BasicBlock::create(cfg);
      auto & tacs = phi_block->tacs();

      // For each incoming basic block, create a new basic block where phi operands are stored
      // All incoming edges get routed through the corresponding intermediate basic block

      // Mapping from original incoming block to intermediate block
      std::unordered_map<ControlFlowGraphNode *, BasicBlock *> intermediateBlocks;

      // Make a copy of the original inEdges to avoid iterator invalidation
      std::vector<ControlFlowGraphEdge *> originalInEdges;
      for (auto & inEdge : phi_block->InEdges())
        originalInEdges.push_back(&inEdge);

      // For each inEdge, route it through a corresponding intermediate block instead
      for (auto inEdge : originalInEdges)
      {
        auto source = inEdge->source();
        BasicBlock * intermediate;

        if (intermediateBlocks.find(source) == intermediateBlocks.end())
        {
          intermediate = BasicBlock::create(cfg);
          intermediate->add_outedge(ass_block);
          intermediateBlocks[source] = intermediate;
        }
        else
        {
          intermediate = intermediateBlocks[source];
        }

        // Re-route this in-edge through the intermediate
        inEdge->divert(intermediate);
      }
      // Finally give the phi_block a single input
      ass_block->add_outedge(phi_block);
      JLM_ASSERT(phi_block->NumInEdges() == 1);

      // For each phi operation, move its operands to the corresponding intermediate blocks instead
      while (tacs.first())
      {
        auto phitac = tacs.first();
        if (!is<SsaPhiOperation>(phitac))
          break;

        const auto phi = static_cast<const SsaPhiOperation *>(&phitac->operation());

        // Instead of having a phi operation, extract its result and give it to an undef operation
        auto phiresult = std::move(phitac->results()[0]);
        auto undef = firstbb->append_first(UndefValueOperation::Create(std::move(phiresult)));

        // Create a mutable variable that will hold the value of the phi result
        auto variable = cfg.module().create_variable(phi->Type());

        JLM_ASSERT(phitac->noperands() == intermediateBlocks.size());
        for (size_t n = 0; n < phitac->noperands(); n++)
        {
          auto incoming = phi->GetIncomingNode(n);
          auto intermediate = intermediateBlocks[incoming];
          JLM_ASSERT(intermediate != nullptr);

          intermediate->append_last(AssignmentOperation::create(phitac->operand(n), variable));
        }

        // In the assignment block, store the variable into the result of the undef operation
        ass_block->append_last(AssignmentOperation::create(variable, undef->result(0)));

        // Remove the phi three address code
        tacs.drop_first();
      }
    }
  };

  auto phi_blocks = collect_phi_blocks(cfg);
  eliminate_phis(cfg, phi_blocks);
}

}

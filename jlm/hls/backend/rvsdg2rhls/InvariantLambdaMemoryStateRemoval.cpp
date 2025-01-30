/*
 * Copyright 2025 Magnus Själander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

void
RemoveInvariantMemoryStateEdges(rvsdg::RegionResult * memoryState)
{
  // We only apply this for memory state edges that is invariant between
  // LambdaEntryMemoryStateSplit and LambdaExitMemoryStateMerge nodes.
  // So we first check if we have a LambdaExitMemoryStateMerge node.
  if (memoryState->origin()->nusers() == 1)
  {
    auto exitNode = rvsdg::output::GetNode(*memoryState->origin());
    if (rvsdg::is<const llvm::LambdaExitMemoryStateMergeOperation>(exitNode->GetOperation()))
    {
      // Check if we have any invariant edge(s) between the two nodes
      std::vector<size_t> indexes;
      std::vector<rvsdg::output *> outputs;
      rvsdg::Node * entryNode = nullptr;
      for (size_t i = 0; i < exitNode->ninputs(); i++)
      {
        // Check if the output has only one user and if it is a LambdaEntryMemoryStateMerge
        if (exitNode->input(i)->origin()->nusers() == 1)
        {
          auto node = rvsdg::output::GetNode(*exitNode->input(i)->origin());
          // TODO change to is<>
          if (jlm::rvsdg::is<const llvm::LambdaEntryMemoryStateSplitOperation>(
                  node->GetOperation()))
          {
            // Found an invariant memory state edge, so going to replace the entryNode
            entryNode = node;
            continue;
          }
        }
        // Keep track of edges that is to be kept since they are not invariant
        outputs.push_back(exitNode->input(i)->origin());
        // Also keep track of the index to be used for diverting edges
        indexes.push_back(i);
      }

      // If the entryNode is not set, then we haven't found any invariant edges
      if (entryNode == nullptr)
      {
        return;
      }

      // Replace LambdaEntryMemoryStateSplit and LambdaExitMemoryStateMerge nodes
      if (outputs.size() == 0)
      {
        // The memory state edge is invariant, so we could in principle remove it from the lambda
        // But the LLVM dialect expects to always have a memory state, so we connect the argument
        // directly to the result
        memoryState->divert_to(entryNode->input(0)->origin());
      }
      else if (outputs.size() == 1)
      {
        // Single edge that is not invariant, so we can elmintate the two MemoryState nodes
        memoryState->divert_to(outputs.front());
        entryNode->output(indexes.front())->divert_users(entryNode->input(0)->origin());
      }
      else
      {
        // Replace the entry and exit node with new ones without the invariant edge(s)
        auto newEntryNodeOutputs = llvm::LambdaEntryMemoryStateSplitOperation::Create(
            *entryNode->input(0)->origin(),
            indexes.size());
        memoryState->divert_to(
            &llvm::LambdaExitMemoryStateMergeOperation::Create(*exitNode->region(), outputs));
        int i = 0;
        for (auto index : indexes)
        {
          entryNode->output(index)->divert_users(newEntryNodeOutputs.at(i));
          i++;
        }
      }
      JLM_ASSERT(exitNode->IsDead());
      rvsdg::remove(exitNode);
      JLM_ASSERT(entryNode->IsDead());
      rvsdg::remove(entryNode);
    }
  }
}

void
RemoveInvariantLambdaMemoryStateEdges(llvm::RvsdgModule & rvsdgModule)
{
  auto & root = rvsdgModule.Rvsdg().GetRootRegion();
  for (auto & node : rvsdg::TopDownTraverser(&root))
  {
    if (rvsdg::is<llvm::LlvmLambdaOperation>(node))
    {
      auto lambda = static_cast<rvsdg::LambdaNode *>(node);

      for (auto result : lambda->subregion()->Results())
      {
        if (jlm::rvsdg::is<const llvm::MemoryStateType>(*result->Type()))
        {
          RemoveInvariantMemoryStateEdges(result);
        }
      }
    }
  }
}

} // namespace jlm::hls

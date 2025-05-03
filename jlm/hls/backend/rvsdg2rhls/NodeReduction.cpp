/*
 * Copyright 2025 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/hls-function-util.hpp>
#include <jlm/hls/backend/rvsdg2rhls/NodeReduction.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rhls-dne.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

/**
 * Tries to reduce multiple memory state split nodes into a single node, and removes unused
 * outputs, which is a result from running the fix_mem_merge() method.
 *
 * @param split_node Node with a llvm::MemoryStateSplitOperation that is to be investigated if it
 * can be reduced
 */
bool
fix_mem_split(rvsdg::Node * split_node)
{
  if (split_node->noutputs() == 1)
  {
    split_node->output(0)->divert_users(split_node->input(0)->origin());
    JLM_ASSERT(split_node->IsDead());
    remove(split_node);
    return true;
  }
  // This merges downward and removes unused outputs (should only exist as a result of eliminating
  // merges)
  std::vector<rvsdg::output *> combined_outputs;
  for (size_t i = 0; i < split_node->noutputs(); ++i)
  {
    if (split_node->output(i)->IsDead())
      continue;
    auto user = get_mem_state_user(split_node->output(i));
    if (TryGetOwnerOp<llvm::MemoryStateSplitOperation>(*user))
    {
      auto sub_split = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*user);
      for (size_t j = 0; j < sub_split->noutputs(); ++j)
      {
        combined_outputs.push_back(sub_split->output(j));
      }
    }
    else
    {
      combined_outputs.push_back(split_node->output(i));
    }
  }
  if (combined_outputs.size() != split_node->noutputs())
  {
    auto new_outputs = llvm::MemoryStateSplitOperation::Create(
        *split_node->input(0)->origin(),
        combined_outputs.size());
    for (size_t i = 0; i < combined_outputs.size(); ++i)
    {
      combined_outputs[i]->divert_users(new_outputs[i]);
    }
    return true;
  }
  return false;
}

/**
 * Removes memory state merge nodes that only have a single state edge, replaces multiple parallel
 * memory state edges with a single state edge, and tries to reduces multiple merge nodes into a
 * single node.
 *
 * @param merge_node Node with a llvm::MemoryStateMergeOperation that is to be investigated if it
 * can be reduced
 */
bool
fix_mem_merge(rvsdg::Node * merge_node)
{
  // Remove single merge
  if (merge_node->ninputs() == 1)
  {
    merge_node->output(0)->divert_users(merge_node->input(0)->origin());
    JLM_ASSERT(merge_node->IsDead());
    remove(merge_node);
    return true;
  }
  std::vector<rvsdg::output *> combined_origins;
  std::unordered_set<rvsdg::SimpleNode *> splits;
  for (size_t i = 0; i < merge_node->ninputs(); ++i)
  {
    auto origin = merge_node->input(i)->origin();
    if (TryGetOwnerOp<llvm::MemoryStateMergeOperation>(*origin))
    {
      auto sub_merge = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*origin);
      for (size_t j = 0; j < sub_merge->ninputs(); ++j)
      {
        combined_origins.push_back(sub_merge->input(j)->origin());
      }
    }
    else if (TryGetOwnerOp<llvm::MemoryStateSplitOperation>(*origin))
    {
      // Ensure that there is only one direct connection to a split.
      // We need to keep one, so that the optimizations for decouple edges work
      auto split = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*origin);
      if (!splits.count(split))
      {
        splits.insert(split);
        combined_origins.push_back(origin);
      }
    }
    else
    {
      combined_origins.push_back(merge_node->input(i)->origin());
    }
  }
  if (combined_origins.empty())
  {
    // If none of the inputs are real keep the first one
    combined_origins.push_back(merge_node->input(0)->origin());
  }
  if (combined_origins.size() != merge_node->ninputs())
  {
    auto new_output = llvm::MemoryStateMergeOperation::Create(combined_origins);
    merge_node->output(0)->divert_users(new_output);
    JLM_ASSERT(merge_node->IsDead());
    return true;
  }
  return false;
}

bool
NodeReduction(rvsdg::Region * sr)
{
  bool any_changed = false;
  bool changed;
  do
  {
    changed = false;
    for (auto & node : rvsdg::BottomUpTraverser(sr))
    {
      if (!node->has_users())
      {
        if (dynamic_cast<const mem_req_op *>(&node->GetOperation()))
        {
          // TODO: fix this once memory connections are explicit
          continue;
        }
        else if (dynamic_cast<const local_mem_req_op *>(&node->GetOperation()))
        {
          continue;
        }
        else if (dynamic_cast<const local_mem_resp_op *>(&node->GetOperation()))
        {
          // TODO: fix - this scenario has only stores and should just be optimized away completely
          continue;
        }
        remove(node);
        changed = true;
      }
      else if (dynamic_cast<rvsdg::LambdaNode *>(node))
      {
        JLM_UNREACHABLE("This function works on lambda subregions");
      }
      else if (auto ln = dynamic_cast<loop_node *>(node))
      {
        changed |= remove_unused_loop_outputs(ln);
        changed |= remove_unused_loop_inputs(ln);
        changed |= remove_unused_loop_backedges(ln);
        changed |= remove_loop_passthrough(ln);
        changed |= dne(ln->subregion());
      }
      else if (dynamic_cast<const llvm::MemoryStateSplitOperation *>(&node->GetOperation()))
      {
        if (fix_mem_split(node))
        {
          changed = true;
          // might break bottom up traversal
          break;
        }
      }
      else if (dynamic_cast<const llvm::MemoryStateMergeOperation *>(&node->GetOperation()))
      {
        if (fix_mem_merge(node))
        {
          changed = true;
          // might break bottom up traversal
          break;
        }
      }
    }
    any_changed |= changed;
  } while (changed);
  return any_changed;
}

void
NodeReduction(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = &graph.GetRootRegion();
  if (root->nnodes() != 1)
  {
    throw util::error("Root should have only one node now");
  }
  auto ln = dynamic_cast<const rvsdg::LambdaNode *>(root->Nodes().begin().ptr());
  if (!ln)
  {
    throw util::error("Node needs to be a lambda");
  }
  NodeReduction(ln->subregion());
}

} // namespace jlm::hls

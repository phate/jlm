/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/hls-function-util.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rhls-dne.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

static bool
remove_unused_loop_backedges(LoopNode * loopNode)
{
  util::HashSet<size_t> resultIndices;
  util::HashSet<size_t> argumentIndices;
  const auto subregion = loopNode->subregion();
  for (const auto argument : subregion->Arguments())
  {
    if ((dynamic_cast<BackEdgeArgument *>(argument) && argument->nusers() == 1)
        || argument->IsDead())
    {
      auto & user = *argument->Users().begin();
      if (const auto result = dynamic_cast<BackEdgeResult *>(&user))
      {
        resultIndices.insert(result->index());
        argumentIndices.insert(argument->index());
      }
    }
  }

  [[maybe_unused]] const auto numRemovedResults = subregion->RemoveResults(resultIndices);
  JLM_ASSERT(numRemovedResults == resultIndices.Size());

  [[maybe_unused]] const auto numRemovedArguments = subregion->RemoveArguments(argumentIndices);
  JLM_ASSERT(numRemovedArguments == argumentIndices.Size());

  return numRemovedArguments != 0;
}

static bool
remove_unused_loop_outputs(LoopNode * ln)
{
  bool any_changed = false;
  // go through in reverse because we remove some
  for (int i = ln->noutputs() - 1; i >= 0; --i)
  {
    const auto out = ln->output(i);
    if (out->nusers() == 0)
    {
      ln->removeLoopOutput(out);
      any_changed = true;
    }
  }
  return any_changed;
}

static bool
remove_loop_passthrough(LoopNode * ln)
{
  bool any_changed = false;
  // go through in reverse because we remove some
  for (int i = ln->ninputs() - 1; i >= 0; --i)
  {
    const auto in = ln->input(i);
    JLM_ASSERT(in->arguments.size() == 1);
    const auto arg = in->arguments.begin();
    if (arg->nusers() != 1)
      continue;

    auto & user = *arg->Users().begin();
    if (const auto result = dynamic_cast<rvsdg::RegionResult *>(&user))
    {
      result->output()->divert_users(in->origin());
      ln->removeLoopOutput(result->output());
      ln->removeLoopInput(arg->input());
      any_changed = true;
    }
  }
  return any_changed;
}

static bool
remove_unused_loop_inputs(LoopNode * ln)
{
  bool any_changed = false;
  auto sr = ln->subregion();
  // go through in reverse because we remove some
  for (int i = ln->ninputs() - 1; i >= 0; --i)
  {
    auto in = ln->input(i);
    JLM_ASSERT(in->arguments.size() == 1);
    auto arg = in->arguments.begin();
    if (arg->nusers() == 0)
    {
      ln->removeLoopInput(in);
      any_changed = true;
    }
  }
  // clean up unused arguments - only ones without an input should be left
  // go through in reverse because we remove some
  for (int i = sr->narguments() - 1; i >= 0; --i)
  {
    auto arg = sr->argument(i);
    if (auto ba = dynamic_cast<BackEdgeArgument *>(arg))
    {
      auto result = ba->result();
      JLM_ASSERT(*result->Type() == *arg->Type());
      if (arg->nusers() == 0 || (arg->nusers() == 1 && result->origin() == arg))
      {
        sr->RemoveResult(result->index());
        sr->RemoveArgument(arg->index());
      }
    }
    else
    {
      JLM_ASSERT(arg->nusers() != 0);
    }
  }
  return any_changed;
}

static bool
dead_spec_gamma(rvsdg::Node * dmux_node)
{
  const auto mux_op = util::assertedCast<const MuxOperation>(&dmux_node->GetOperation());
  JLM_ASSERT(mux_op->discarding);
  // check if all inputs have the same origin
  bool all_inputs_same = true;
  auto first_origin = dmux_node->input(1)->origin();
  for (size_t i = 2; i < dmux_node->ninputs(); ++i)
  {
    if (dmux_node->input(i)->origin() != first_origin)
    {
      all_inputs_same = false;
      break;
    }
  }
  if (all_inputs_same)
  {
    dmux_node->output(0)->divert_users(first_origin);
    remove(dmux_node);
    return true;
  }
  return false;
}

static bool
dead_nonspec_gamma(rvsdg::Node * ndmux_node)
{
  auto mux_op = util::assertedCast<const MuxOperation>(&ndmux_node->GetOperation());
  JLM_ASSERT(!mux_op->discarding);
  // check if all inputs go to outputs of same branch
  bool all_inputs_same_branch = true;
  rvsdg::Node * origin_branch = nullptr;
  for (size_t i = 1; i < ndmux_node->ninputs(); ++i)
  {
    if (auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*ndmux_node->input(i)->origin()))
    {
      if (dynamic_cast<const BranchOperation *>(&node->GetOperation())
          && ndmux_node->input(i)->origin()->nusers() == 1)
      {
        if (i == 1)
        {
          origin_branch = node;
          continue;
        }
        else if (origin_branch == node)
        {
          continue;
        }
      }
    }
    all_inputs_same_branch = false;
    break;
  }
  if (all_inputs_same_branch && origin_branch->input(0)->origin() == ndmux_node->input(0)->origin())
  {
    // same control origin + all inputs to branch
    ndmux_node->output(0)->divert_users(origin_branch->input(1)->origin());
    remove(ndmux_node);
    JLM_ASSERT(origin_branch != nullptr);
    remove(origin_branch);
    return true;
  }
  return false;
}

static bool
dead_loop(rvsdg::Node * ndmux_node)
{
  const auto mux_op = util::assertedCast<const MuxOperation>(&ndmux_node->GetOperation());
  JLM_ASSERT(!mux_op->discarding);
  // origin is a backedege argument
  auto backedge_arg = dynamic_cast<BackEdgeArgument *>(ndmux_node->input(2)->origin());
  if (!backedge_arg)
  {
    return false;
  }
  // one branch
  if (ndmux_node->output(0)->nusers() != 1)
  {
    return false;
  }
  auto branch_in_node =
      rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*ndmux_node->output(0)->Users().begin());
  if (!branch_in_node || !dynamic_cast<const BranchOperation *>(&branch_in_node->GetOperation()))
  {
    return false;
  }
  // one buffer
  if (branch_in_node->output(1)->nusers() != 1)
  {
    return false;
  }
  auto buf_in_node =
      rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*branch_in_node->output(1)->Users().begin());
  if (!buf_in_node || !dynamic_cast<const BufferOperation *>(&buf_in_node->GetOperation()))
  {
    return false;
  }
  auto buf_out = buf_in_node->output(0);
  if (buf_out != backedge_arg->result()->origin())
  {
    // no connection back up
    return false;
  }
  // depend on same control
  auto branch_cond_origin = branch_in_node->input(0)->origin();
  auto pred_buf_out_node =
      rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*ndmux_node->input(0)->origin());
  if (!pred_buf_out_node
      || !dynamic_cast<const PredicateBufferOperation *>(&pred_buf_out_node->GetOperation()))
  {
    return false;
  }
  auto pred_buf_cond_origin = pred_buf_out_node->input(0)->origin();
  // TODO: remove this once predicate buffers decouple combinatorial loops
  auto extra_buf_out_node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*pred_buf_cond_origin);
  if (!extra_buf_out_node
      || !dynamic_cast<const BufferOperation *>(&extra_buf_out_node->GetOperation()))
  {
    return false;
  }
  auto extra_buf_cond_origin = extra_buf_out_node->input(0)->origin();

  if (auto pred_be = dynamic_cast<BackEdgeArgument *>(extra_buf_cond_origin))
  {
    extra_buf_cond_origin = pred_be->result()->origin();
  }
  if (extra_buf_cond_origin != branch_cond_origin)
  {
    return false;
  }
  // divert users
  branch_in_node->output(0)->divert_users(ndmux_node->input(1)->origin());
  buf_out->divert_users(backedge_arg);
  remove(buf_in_node);
  remove(branch_in_node);
  auto region = ndmux_node->region();
  remove(ndmux_node);
  region->RemoveResult(backedge_arg->result()->index());
  region->RemoveArgument(backedge_arg->index());
  return true;
}

static bool
dead_loop_lcb(rvsdg::Node * lcb_node)
{
  JLM_ASSERT(jlm::rvsdg::is<LoopConstantBufferOperation>(lcb_node));

  // one branch
  if (lcb_node->output(0)->nusers() != 1)
  {
    return false;
  }
  auto [branchNode, branchOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<BranchOperation>(*lcb_node->output(0)->Users().begin());
  if (!branchNode || !branchOperation || !branchOperation->loop)
  {
    return false;
  }
  // no user
  if (branchNode->output(1)->nusers())
  {
    return false;
  }
  // depend on same control
  auto branch_cond_origin = branchNode->input(0)->origin();
  auto pred_buf_out = dynamic_cast<rvsdg::NodeOutput *>(lcb_node->input(0)->origin());
  if (!pred_buf_out
      || !dynamic_cast<const PredicateBufferOperation *>(&pred_buf_out->node()->GetOperation()))
  {
    return false;
  }
  auto pred_buf_cond_origin = pred_buf_out->node()->input(0)->origin();
  // TODO: remove this once predicate buffers decouple combinatorial loops
  auto extra_buf_out = dynamic_cast<rvsdg::NodeOutput *>(pred_buf_cond_origin);
  if (!extra_buf_out
      || !dynamic_cast<const BufferOperation *>(&extra_buf_out->node()->GetOperation()))
  {
    return false;
  }
  auto extra_buf_cond_origin = extra_buf_out->node()->input(0)->origin();

  if (auto pred_be = dynamic_cast<BackEdgeArgument *>(extra_buf_cond_origin))
  {
    extra_buf_cond_origin = pred_be->result()->origin();
  }
  if (extra_buf_cond_origin != branch_cond_origin)
  {
    return false;
  }
  // divert users
  branchNode->output(0)->divert_users(lcb_node->input(1)->origin());
  remove(branchNode);
  remove(lcb_node);
  return true;
}

static bool
fix_mem_split(rvsdg::Node * split_node)
{
  if (split_node->noutputs() == 1)
  {
    split_node->output(0)->divert_users(split_node->input(0)->origin());
    JLM_ASSERT(split_node->IsDead());
    remove(split_node);
    return true;
  }
  // this merges downward and removes unused outputs (should only exist as a result of eliminating
  // merges)
  std::vector<rvsdg::Output *> combined_outputs;
  for (size_t i = 0; i < split_node->noutputs(); ++i)
  {
    if (split_node->output(i)->IsDead())
      continue;
    auto user = get_mem_state_user(split_node->output(i));
    if (rvsdg::IsOwnerNodeOperation<llvm::MemoryStateSplitOperation>(*user))
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

static bool
fix_mem_merge(rvsdg::Node * merge_node)
{
  // remove single merge
  if (merge_node->ninputs() == 1)
  {
    merge_node->output(0)->divert_users(merge_node->input(0)->origin());
    JLM_ASSERT(merge_node->IsDead());
    remove(merge_node);
    return true;
  }
  std::vector<rvsdg::Output *> combined_origins;
  std::unordered_set<rvsdg::SimpleNode *> splits;
  for (size_t i = 0; i < merge_node->ninputs(); ++i)
  {
    auto origin = merge_node->input(i)->origin();
    if (rvsdg::IsOwnerNodeOperation<llvm::MemoryStateMergeOperation>(*origin))
    {
      auto sub_merge = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*origin);
      for (size_t j = 0; j < sub_merge->ninputs(); ++j)
      {
        combined_origins.push_back(sub_merge->input(j)->origin());
      }
    }
    else if (rvsdg::IsOwnerNodeOperation<llvm::MemoryStateSplitOperation>(*origin))
    {
      // ensure that there is only one direct connection to a split.
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
    // if none of the inputs are real keep the first one
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
RhlsDeadNodeElimination::Run(
    rvsdg::Region & region,
    util::StatisticsCollector & statisticsCollector)
{
  bool any_changed = false;
  bool changed = false;
  do
  {
    changed = false;
    for (auto & node : rvsdg::BottomUpTraverser(&region))
    {
      if (node->IsDead())
      {
        if (rvsdg::is<MemoryRequestOperation>(node))
        {
          // TODO: fix this once memory connections are explicit
          continue;
        }
        if (rvsdg::is<LocalMemoryRequestOperation>(node))
        {
          continue;
        }
        if (rvsdg::is<LocalMemoryResponseOperation>(node))
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
      else if (auto ln = dynamic_cast<LoopNode *>(node))
      {
        changed |= remove_unused_loop_outputs(ln);
        changed |= remove_unused_loop_inputs(ln);
        changed |= remove_unused_loop_backedges(ln);
        changed |= remove_loop_passthrough(ln);
        changed |= Run(*ln->subregion(), statisticsCollector);
      }
      else if (const auto mux = dynamic_cast<const MuxOperation *>(&node->GetOperation()))
      {
        if (mux->discarding)
        {
          changed |= dead_spec_gamma(node);
        }
        else
        {
          changed |= dead_nonspec_gamma(node) || dead_loop(node);
        }
      }
      else if (rvsdg::is<LoopConstantBufferOperation>(node))
      {
        changed |= dead_loop_lcb(node);
      }
      else if (dynamic_cast<const llvm::MemoryStateSplitOperation *>(&node->GetOperation()))
      {
        if (fix_mem_split(node))
        {
          changed = true;
        }
      }
      else if (dynamic_cast<const llvm::MemoryStateMergeOperation *>(&node->GetOperation()))
      {
        if (fix_mem_merge(node))
        {
          changed = true;
        }
      }
      if (changed)
      {
        // Changes might break bottom up traversal
        break;
      }
    }
    any_changed |= changed;
  } while (changed);

  return any_changed;
}

RhlsDeadNodeElimination::~RhlsDeadNodeElimination() noexcept = default;

RhlsDeadNodeElimination::RhlsDeadNodeElimination()
    : Transformation("RhlsDeadNodeElimination")
{}

void
RhlsDeadNodeElimination::Run(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  auto & graph = rvsdgModule.Rvsdg();
  const auto rootRegion = &graph.GetRootRegion();
  if (rootRegion->numNodes() != 1)
  {
    throw util::Error("Root should have only one node now");
  }
  const auto lambdaNode =
      dynamic_cast<const rvsdg::LambdaNode *>(rootRegion->Nodes().begin().ptr());
  if (!lambdaNode)
  {
    throw util::Error("Node needs to be a lambda");
  }
  Run(*lambdaNode->subregion(), statisticsCollector);
}

} // namespace jlm::hls

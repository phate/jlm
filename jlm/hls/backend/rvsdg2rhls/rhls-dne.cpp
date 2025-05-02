/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/rhls-dne.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

bool
remove_unused_loop_backedges(loop_node * ln)
{
  bool any_changed = false;
  auto sr = ln->subregion();
  // go through in reverse because we remove some
  for (int i = sr->narguments() - 1; i >= 0; --i)
  {
    auto arg = sr->argument(i);
    if (dynamic_cast<backedge_argument *>(arg) && arg->nusers() == 1)
    {
      auto user = *arg->begin();
      if (auto result = dynamic_cast<backedge_result *>(user))
      {
        sr->RemoveResult(result->index());
        sr->RemoveArgument(arg->index());
        any_changed = true;
      }
    }
  }
  return any_changed;
}

bool
remove_unused_loop_outputs(loop_node * ln)
{
  bool any_changed = false;
  auto sr = ln->subregion();
  // go through in reverse because we remove some
  for (int i = ln->noutputs() - 1; i >= 0; --i)
  {
    auto out = ln->output(i);
    if (out->nusers() == 0)
    {
      JLM_ASSERT(out->results.size() == 1);
      auto result = out->results.begin();
      sr->RemoveResult(result->index());
      ln->RemoveOutput(out->index());
      any_changed = true;
    }
  }
  return any_changed;
}

bool
remove_loop_passthrough(loop_node * ln)
{
  bool any_changed = false;
  auto sr = ln->subregion();
  // go through in reverse because we remove some
  for (int i = ln->ninputs() - 1; i >= 0; --i)
  {
    auto in = ln->input(i);
    JLM_ASSERT(in->arguments.size() == 1);
    auto arg = in->arguments.begin();
    if (arg->nusers() == 1)
    {
      auto user = *arg->begin();
      if (auto result = dynamic_cast<rvsdg::RegionResult *>(user))
      {
        auto out = result->output();
        out->divert_users(in->origin());
        sr->RemoveResult(result->index());
        ln->RemoveOutput(out->index());
        auto inputIndex = arg->input()->index();
        sr->RemoveArgument(arg->index());
        ln->RemoveInput(inputIndex);
        any_changed = true;
      }
    }
  }
  return any_changed;
}

bool
remove_unused_loop_inputs(loop_node * ln)
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
      sr->RemoveArgument(arg->index());
      ln->RemoveInput(in->index());
      any_changed = true;
    }
  }
  // clean up unused arguments - only ones without an input should be left
  // go through in reverse because we remove some
  for (int i = sr->narguments() - 1; i >= 0; --i)
  {
    auto arg = sr->argument(i);
    if (auto ba = dynamic_cast<backedge_argument *>(arg))
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

bool
dead_spec_gamma(rvsdg::SimpleNode * dmux_node)
{
  auto mux_op = dynamic_cast<const jlm::hls::mux_op *>(&dmux_node->GetOperation());
  JLM_ASSERT(mux_op);
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

bool
dead_nonspec_gamma(rvsdg::SimpleNode * ndmux_node)
{
  auto mux_op = dynamic_cast<const hls::mux_op *>(&ndmux_node->GetOperation());
  JLM_ASSERT(mux_op);
  JLM_ASSERT(!mux_op->discarding);
  // check if all inputs go to outputs of same branch
  bool all_inputs_same_branch = true;
  rvsdg::Node * origin_branch = nullptr;
  for (size_t i = 1; i < ndmux_node->ninputs(); ++i)
  {
    if (auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*ndmux_node->input(i)->origin()))
    {
      if (dynamic_cast<const branch_op *>(&node->GetOperation())
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

bool
dead_loop(rvsdg::SimpleNode * ndmux_node)
{
  auto mux_op = dynamic_cast<const hls::mux_op *>(&ndmux_node->GetOperation());
  JLM_ASSERT(mux_op);
  JLM_ASSERT(!mux_op->discarding);
  // origin is a backedege argument
  auto backedge_arg = dynamic_cast<backedge_argument *>(ndmux_node->input(2)->origin());
  if (!backedge_arg)
  {
    return false;
  }
  // one branch
  if (ndmux_node->output(0)->nusers() != 1)
  {
    return false;
  }
  auto branch_in_node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(**ndmux_node->output(0)->begin());
  if (!branch_in_node || !dynamic_cast<const branch_op *>(&branch_in_node->GetOperation()))
  {
    return false;
  }
  // one buffer
  if (branch_in_node->output(1)->nusers() != 1)
  {
    return false;
  }
  auto buf_in_node =
      rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(**branch_in_node->output(1)->begin());
  if (!buf_in_node || !dynamic_cast<const buffer_op *>(&buf_in_node->GetOperation()))
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
      || !dynamic_cast<const predicate_buffer_op *>(&pred_buf_out_node->GetOperation()))
  {
    return false;
  }
  auto pred_buf_cond_origin = pred_buf_out_node->input(0)->origin();
  // TODO: remove this once predicate buffers decouple combinatorial loops
  auto extra_buf_out_node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*pred_buf_cond_origin);
  if (!extra_buf_out_node || !dynamic_cast<const buffer_op *>(&extra_buf_out_node->GetOperation()))
  {
    return false;
  }
  auto extra_buf_cond_origin = extra_buf_out_node->input(0)->origin();

  if (auto pred_be = dynamic_cast<backedge_argument *>(extra_buf_cond_origin))
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

bool
dne(rvsdg::Region * sr)
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
        if (auto simpleNode = dynamic_cast<const rvsdg::SimpleNode *>(node))
        {
          if (dynamic_cast<const mem_req_op *>(&simpleNode->GetOperation()))
          {
            // TODO: fix this once memory connections are explicit
            continue;
          }
          else if (dynamic_cast<const local_mem_req_op *>(&simpleNode->GetOperation()))
          {
            continue;
          }
          else if (dynamic_cast<const local_mem_resp_op *>(&simpleNode->GetOperation()))
          {
            // TODO: fix - this scenario has only stores and should just be optimized away
            // completely
            continue;
          }
        }
        remove(node);
        changed = true;
      }
      else if (auto ln = dynamic_cast<loop_node *>(node))
      {
        changed |= remove_unused_loop_outputs(ln);
        changed |= remove_unused_loop_inputs(ln);
        changed |= remove_unused_loop_backedges(ln);
        changed |= remove_loop_passthrough(ln);
        changed |= dne(ln->subregion());
      }
      else if (auto simpleNode = dynamic_cast<rvsdg::SimpleNode *>(node))
      {
        if (auto mux = dynamic_cast<const mux_op *>(&simpleNode->GetOperation()))
        {
          if (mux->discarding)
          {
            changed |= dead_spec_gamma(simpleNode);
          }
          else
          {
            changed |= dead_nonspec_gamma(simpleNode) || dead_loop(simpleNode);
          }
        }
      }
    }
    any_changed |= changed;
  } while (changed);
  return any_changed;
}

void
dne(llvm::RvsdgModule & rm)
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
  dne(ln->subregion());
}

} // namespace jlm::hls

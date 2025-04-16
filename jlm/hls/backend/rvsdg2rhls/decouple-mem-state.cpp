/*
 * Copyright 2025 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/decouple-mem-state.hpp>
#include <jlm/hls/backend/rvsdg2rhls/hls-function-util.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-conv.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-sep.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rhls-dne.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/hls/util/view.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/view.hpp>

#include <deque>

namespace jlm::hls
{

bool
is_store(rvsdg::SimpleNode * node)
{
  return dynamic_cast<const llvm::StoreNonVolatileOperation *>(&node->GetOperation());
}

bool
is_load(rvsdg::SimpleNode * node)
{
  return dynamic_cast<const llvm::LoadNonVolatileOperation *>(&node->GetOperation());
}

rvsdg::output *
follow_state_edge(
    rvsdg::input * state_edge,
    std::vector<rvsdg::SimpleNode *> & mem_ops,
    bool modify);

rvsdg::output *
trace_edge(
    rvsdg::input * state_edge,
    rvsdg::output * new_edge,
    rvsdg::SimpleNode * target_call,
    rvsdg::output * end)
{
  rvsdg::input * previous_state_edge = nullptr;
  while (true)
  {
    // make sure we make progress
    JLM_ASSERT(previous_state_edge != state_edge);
    previous_state_edge = state_edge;
    // terminate once desired point is reached
    if (state_edge->origin() == end)
    {
      return end;
    }
    auto new_edge_user = get_mem_state_user(new_edge);
    JLM_ASSERT(new_edge_user->region() == state_edge->region());
    if (auto rr = dynamic_cast<jlm::rvsdg::RegionResult *>(state_edge))
    {
      return rr->output();
    }
    else if (auto ln = rvsdg::TryGetOwnerNode<loop_node>(*state_edge))
    {
      auto si = util::AssertedCast<rvsdg::StructuralInput>(state_edge);
      auto arg = si->arguments.begin().ptr();
      std::vector<rvsdg::SimpleNode *> mem_ops;
      auto out = follow_state_edge(get_mem_state_user(arg), mem_ops, false);
      if (std::count(mem_ops.begin(), mem_ops.end(), target_call))
      {
        // only route new edge through if target is contained
        auto new_out = ln->AddLoopVar(new_edge);
        new_edge_user->divert_to(new_out);
        auto new_in = util::AssertedCast<rvsdg::StructuralInput>(get_mem_state_user(new_edge));
        JLM_ASSERT(
            out
            == trace_edge(
                get_mem_state_user(arg),
                new_in->arguments.begin().ptr(),
                target_call,
                end));
        new_edge = new_out;
        convert_loop_state_to_lcb(new_in);
      }
      JLM_ASSERT(TryGetOwnerOp<loop_op>(*out));
      JLM_ASSERT(out->region() == state_edge->region());
      state_edge = get_mem_state_user(out);
      continue;
    }
    auto si = util::AssertedCast<rvsdg::SimpleInput>(state_edge);
    auto sn = si->node();
    auto new_si = util::AssertedCast<rvsdg::SimpleInput>(new_edge_user);
    auto new_sn = new_si->node();
    auto br = TryGetOwnerOp<branch_op>(*state_edge);
    auto mux = TryGetOwnerOp<mux_op>(*state_edge);
    if (br && !br->loop) // this is an example of why preserving structural nodes would be nice
    {
      // start of gamma
      auto nbr = branch_op::create(*sn->input(0)->origin(), *new_edge);
      auto nmux = mux_op::create(*sn->input(0)->origin(), nbr, false)[0];
      new_edge_user->divert_to(nmux);
      rvsdg::output * out = nullptr;
      for (size_t i = 0; i < sn->noutputs(); ++i)
      {
        out = trace_edge(get_mem_state_user(sn->output(i)), nbr[i], target_call, end);
      }
      JLM_ASSERT(out);
      JLM_ASSERT(TryGetOwnerOp<mux_op>(*out));
      state_edge = get_mem_state_user(out);
      new_edge = nmux;
    }
    else if (br)
    {
      // end of loop
      JLM_ASSERT(TryGetOwnerOp<branch_op>(*new_edge_user));
      JLM_ASSERT(br->loop);
      return util::AssertedCast<rvsdg::RegionResult>(get_mem_state_user(sn->output(0)))->output();
    }
    else if (mux && !mux->loop)
    {
      // end of gamma
      JLM_ASSERT(TryGetOwnerOp<mux_op>(*new_edge_user));
      return sn->output(0);
    }
    else if (mux)
    {
      // start of theta
      JLM_ASSERT(TryGetOwnerOp<mux_op>(*new_edge_user));
      JLM_ASSERT(mux->loop);
      state_edge = get_mem_state_user(sn->output(0));
      new_edge = new_sn->output(0);
    }
    else if (TryGetOwnerOp<loop_constant_buffer_op>(*state_edge))
    {
      // start of loop
      JLM_ASSERT(TryGetOwnerOp<mux_op>(*new_edge_user));
      state_edge = get_mem_state_user(sn->output(0));
      new_edge = new_sn->output(0);
    }
    else if (TryGetOwnerOp<llvm::MemoryStateSplitOperation>(*state_edge))
    {
      rvsdg::output * after_merge = nullptr;
      for (size_t i = 0; i < sn->noutputs(); ++i)
      {
        // pick right edge by searching for target
        std::vector<rvsdg::SimpleNode *> mem_ops;
        after_merge = follow_state_edge(get_mem_state_user(sn->output(i)), mem_ops, false);
        JLM_ASSERT(after_merge);
        JLM_ASSERT(TryGetOwnerOp<llvm::MemoryStateMergeOperation>(*after_merge));
        if (std::count(mem_ops.begin(), mem_ops.end(), target_call))
        {
          auto out = trace_edge(get_mem_state_user(sn->output(i)), new_edge, target_call, end);
          JLM_ASSERT(out == after_merge);
        }
        else
        {
          // nothing relevant below the split - can just ignore it
        }
      }
      state_edge = get_mem_state_user(after_merge);
      new_edge = new_edge_user->origin();
    }
    else if (
        TryGetOwnerOp<llvm::MemoryStateMergeOperation>(*state_edge)
        || TryGetOwnerOp<llvm::LambdaExitMemoryStateMergeOperation>(*state_edge))
    {
      // we did not split the new state
      return sn->output(0);
    }
    else if (TryGetOwnerOp<state_gate_op>(*state_edge))
    {
      state_edge = get_mem_state_user(sn->output(si->index()));
    }
    else if (TryGetOwnerOp<llvm::LoadNonVolatileOperation>(*state_edge))
    {
      state_edge = get_mem_state_user(sn->output(1));
    }
    else if (TryGetOwnerOp<llvm::CallOperation>(*state_edge))
    {
      auto state_origin = state_edge->origin();
      if (sn == target_call)
      {
        // move decouple call to new edge
        state_edge = get_mem_state_user(sn->output(sn->noutputs() - 1));
        state_edge->divert_to(state_origin);
        si->divert_to(new_edge);
        new_edge_user->divert_to(sn->output(sn->noutputs() - 1));
        new_edge = new_edge_user->origin();
      }
      else
      {
        state_edge = get_mem_state_user(sn->output(sn->noutputs() - 1));
      }
    }
    else
    {
      JLM_UNREACHABLE("whoops");
    }
  }
}

void
handle_structural(
    std::vector<std::tuple<rvsdg::SimpleNode *, rvsdg::input *>> & outstanding_dec_reqs,
    std::vector<rvsdg::SimpleNode *> & mem_ops,
    rvsdg::input * state_edge_before,
    rvsdg::output * state_edge_after)
{
  JLM_ASSERT(state_edge_before->region() == state_edge_after->region());
  // the reqs we encountered
  for (auto op : mem_ops)
  {
    if (is_dec_req(op))
    {
      outstanding_dec_reqs.push_back(std::make_tuple(op, state_edge_before));
    }
  }
  for (auto op : mem_ops)
  {
    if (is_store(op))
    {
      // can't handle things if there is a store on the edge in the same loop/gamma
      return;
    }
  }
  for (auto resp : mem_ops)
  {
    if (!is_dec_res(resp))
      continue;
    auto res_constant = trace_constant(resp->input(1)->origin());
    JLM_ASSERT(res_constant);
    for (auto [req, state_edge_req] : outstanding_dec_reqs)
    {
      auto req_constant = trace_constant(req->input(1)->origin());
      if (*req_constant == *res_constant)
      {
        // we found a match and can split the req off the state edge between input and output
        // in this area both the req and resp can run separate from the main edge
        auto split_outputs = llvm::MemoryStateSplitOperation::Create(*state_edge_req->origin(), 3);
        state_edge_req->divert_to(split_outputs[0]);
        JLM_ASSERT(state_edge_after->region() == split_outputs[1]->region());
        auto after_user = get_mem_state_user(state_edge_after);
        std::vector<rvsdg::output *> operands(
            { state_edge_after, split_outputs[1], split_outputs[2] });
        auto merge_out = llvm::MemoryStateMergeOperation::Create(operands);
        after_user->divert_to(merge_out);
        trace_edge(state_edge_req, split_outputs[1], req, state_edge_after);
        trace_edge(state_edge_req, split_outputs[2], resp, state_edge_after);
      }
    }
  }
}

void
optimize_single_mem_op_loop(
    std::vector<rvsdg::SimpleNode *> & mem_ops,
    rvsdg::input * state_edge_before,
    rvsdg::output * state_edge_after)
{
  // the idea here is that if there is only one memory operation, and no other memory
  // operations/staet gates on a state edge in a loop we can remove the backedge part and treat the
  // edge like a loop constant, albeit with an output this will especially be important for stores
  // that have a response.
  // TODO: should this also be enabled for just memory state gates?
  if (mem_ops.size() == 1 && (is_store(mem_ops[0]) || is_load(mem_ops[0])))
  {
    // before and after belong to same loop node
    JLM_ASSERT(rvsdg::TryGetOwnerNode<loop_node>(*state_edge_before));
    JLM_ASSERT(
        rvsdg::TryGetOwnerNode<loop_node>(*state_edge_before)
        == rvsdg::TryGetOwnerNode<loop_node>(*state_edge_after));
    convert_loop_state_to_lcb(state_edge_before);
  }
}

rvsdg::output *
follow_state_edge(
    rvsdg::input * state_edge,
    std::vector<rvsdg::SimpleNode *> & mem_ops,
    bool modify)
{
  // we use input so we can handle the scenario of a store having an extra user to deq from addrq
  /*
    things we can encounter:
    * region result:
        * return associated output
    * loop-node:
        * follow subregion
        * check which subset of dec_req, dec_resp and store is contained using handle_structural
    * converted gamma:
        * need to follow all sub-paths
        * begins with branch with !br->loop
            * follow_state_edge
        * ends with mux
            * return mux output
        * check which subset of dec_req, dec_resp and store is contained using handle_structural
    * mem state split/merge:
        * follow_state_edge/return merge output
    * load/state-gate
        * continue on state output
    * store/call
        * add to mem_ops
        * continue on state output
        * special case for store - can have multiple users because of addr_deq
  */
  // this tracks decouple requests that have not been handled yet
  std::vector<std::tuple<rvsdg::SimpleNode *, rvsdg::input *>> outstanding_dec_reqs;
  while (true)
  {
    if (auto rr = dynamic_cast<jlm::rvsdg::RegionResult *>(state_edge))
    {
      return rr->output();
    }
    else if (rvsdg::TryGetOwnerNode<loop_node>(*state_edge))
    {
      std::vector<rvsdg::SimpleNode *> loop_mem_ops;
      auto si = jlm::util::AssertedCast<rvsdg::StructuralInput>(state_edge);
      auto arg = si->arguments.begin().ptr();
      auto out = follow_state_edge(get_mem_state_user(arg), loop_mem_ops, modify);
      // get this here before the graph is modified by handle_structural
      auto new_state_edge = get_mem_state_user(out);
      if (modify)
      {
        handle_structural(outstanding_dec_reqs, loop_mem_ops, state_edge, out);
        optimize_single_mem_op_loop(loop_mem_ops, state_edge, out);
      }
      state_edge = new_state_edge;
      mem_ops.insert(mem_ops.cend(), loop_mem_ops.begin(), loop_mem_ops.end());
      continue;
    }
    auto si = jlm::util::AssertedCast<rvsdg::SimpleInput>(state_edge);
    auto sn = si->node();
    auto br = TryGetOwnerOp<branch_op>(*state_edge);
    auto mux = TryGetOwnerOp<mux_op>(*state_edge);
    if (br && !br->loop) // this is an example of why preserving structural nodes would be nice
    {
      std::vector<rvsdg::SimpleNode *> gamma_mem_ops;
      // start of gamma
      rvsdg::output * out = nullptr;
      for (size_t i = 0; i < sn->noutputs(); ++i)
      {
        out = follow_state_edge(get_mem_state_user(sn->output(i)), gamma_mem_ops, modify);
      }
      JLM_ASSERT(TryGetOwnerOp<mux_op>(*out));
      JLM_ASSERT(out);
      // get this here before the graph is modified by handle_structural
      auto new_state_edge = get_mem_state_user(out);
      if (modify)
        handle_structural(outstanding_dec_reqs, gamma_mem_ops, state_edge, out);
      state_edge = new_state_edge;
      mem_ops.insert(mem_ops.cend(), gamma_mem_ops.begin(), gamma_mem_ops.end());
    }
    else if (br)
    {
      // end of loop
      JLM_ASSERT(br->loop);
      return util::AssertedCast<rvsdg::RegionResult>(get_mem_state_user(sn->output(0)))->output();
    }
    else if (mux && !mux->loop)
    {
      // end of gamma
      return sn->output(0);
    }
    else if (mux)
    {
      // start of theta
      JLM_ASSERT(mux->loop);
      state_edge = get_mem_state_user(sn->output(0));
    }
    else if (TryGetOwnerOp<loop_constant_buffer_op>(*state_edge))
    {
      // start of theta
      state_edge = get_mem_state_user(sn->output(0));
    }
    else if (
        TryGetOwnerOp<llvm::MemoryStateSplitOperation>(*state_edge)
        || TryGetOwnerOp<llvm::LambdaEntryMemoryStateSplitOperation>(*state_edge))
    {
      for (size_t i = 0; i < sn->noutputs(); ++i)
      {
        auto followed = follow_state_edge(get_mem_state_user(sn->output(i)), mem_ops, modify);
        JLM_ASSERT(followed);
        JLM_ASSERT(
            TryGetOwnerOp<llvm::MemoryStateMergeOperation>(*followed)
            || TryGetOwnerOp<llvm::LambdaExitMemoryStateMergeOperation>(*followed));
        state_edge = get_mem_state_user(followed);
      }
    }
    else if (
        TryGetOwnerOp<llvm::MemoryStateMergeOperation>(*state_edge)
        || TryGetOwnerOp<llvm::LambdaExitMemoryStateMergeOperation>(*state_edge))
    {
      return sn->output(0);
    }
    else if (TryGetOwnerOp<state_gate_op>(*state_edge))
    {
      mem_ops.push_back(sn);
      state_edge = get_mem_state_user(sn->output(si->index()));
    }
    else if (TryGetOwnerOp<llvm::LoadNonVolatileOperation>(*state_edge))
    {
      mem_ops.push_back(sn);
      state_edge = get_mem_state_user(sn->output(1));
    }
    else if (TryGetOwnerOp<llvm::CallOperation>(*state_edge))
    {
      mem_ops.push_back(sn);
      state_edge = get_mem_state_user(sn->output(sn->noutputs() - 1));
    }
    else if (TryGetOwnerOp<llvm::StoreNonVolatileOperation>(*state_edge))
    {
      mem_ops.push_back(sn);
      JLM_ASSERT(sn->output(0)->nusers() == 1);
      state_edge = get_mem_state_user(sn->output(0));
      // handle case of store that has one edge going off to deq
      if (TryGetOwnerOp<llvm::MemoryStateSplitOperation>(*state_edge))
      {
        // output 0 is the normal edge
        state_edge =
            get_mem_state_user(rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*state_edge)->output(0));
      }
    }
    else
    {
      JLM_UNREACHABLE("whoops");
    }
  }
}

void
convert_loop_state_to_lcb(rvsdg::input * loop_state_input)
{
  JLM_ASSERT(rvsdg::is<rvsdg::StateType>(loop_state_input->type()));
  JLM_ASSERT(rvsdg::TryGetOwnerNode<loop_node>(*loop_state_input));
  auto si = util::AssertedCast<rvsdg::StructuralInput>(loop_state_input);
  auto arg = si->arguments.begin().ptr();
  auto user = get_mem_state_user(arg);
  auto mux = TryGetOwnerOp<mux_op>(*user);
  JLM_ASSERT(mux && mux->loop);
  auto mux_node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*user);
  auto lcb_out = loop_constant_buffer_op::create(
      *mux_node->input(0)->origin(),
      *mux_node->input(1)->origin())[0];
  mux_node->output(0)->divert_users(lcb_out);
  JLM_ASSERT(mux_node->IsDead());
  remove(mux_node);
}

void
decouple_mem_state(rvsdg::Region * region)
{
  auto lambda = util::AssertedCast<const jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
  auto state_arg = GetMemoryStateArgument(*lambda);
  if (!state_arg)
  {
    // No memstate, i.e., no memory used
    return;
  }
  //  * pass after mem-queue
  //      * common edge is edge 0 in splits
  //      * keep response on common edge
  //          * seperate out request
  //      * [ ] scenario 1 - accross outer loops
  //          * split off before loop with req, merge after loop with resp
  //      * [ ] scenario 2.1 - within outer loop - no store on edge in loop
  //      * split before, merge after loop
  //      * [ ] scenario 2.2 - within outer loop - store on edge in loop
  //      * split at highest loop that contains no store on edge
  //          * store not being at higher level doesn't work
  //  * apply recursively - i.e. the same way for inner loops as for outer
  auto state_user = get_mem_state_user(state_arg);
  auto entry_op = TryGetOwnerOp<llvm::LambdaEntryMemoryStateSplitOperation>(*state_user);
  JLM_ASSERT(entry_op);
  auto entry_node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*state_user);
  JLM_ASSERT(entry_node);
  auto state_res = GetMemoryStateResult(*lambda);
  auto exit_op = TryGetOwnerOp<llvm::LambdaExitMemoryStateMergeOperation>(*state_res->origin());
  auto exit_node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*state_res->origin());
  JLM_ASSERT(exit_node);
  JLM_ASSERT(exit_op);
  JLM_ASSERT(entry_node->noutputs() == exit_node->ninputs());
  // process different pointer arg edges separately
  for (size_t i = 0; i < entry_node->noutputs(); ++i)
  {
    std::vector<rvsdg::SimpleNode *> mem_ops;
    std::vector<std::tuple<rvsdg::SimpleNode *, rvsdg::input *>> dummy;
    follow_state_edge(get_mem_state_user(entry_node->output(i)), mem_ops, true);
    // we need this one final time across the whole lambda - at least for this edge
    handle_structural(
        dummy,
        mem_ops,
        get_mem_state_user(entry_node->output(i)),
        exit_node->input(i)->origin());
  }

  dne(lambda->subregion());
}

void
decouple_mem_state(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = &graph.GetRootRegion();
  decouple_mem_state(root);
}
}

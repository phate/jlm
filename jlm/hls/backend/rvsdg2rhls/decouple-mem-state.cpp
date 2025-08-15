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
#include <jlm/llvm/ir/LambdaMemoryState.hpp>
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

#include <algorithm>
#include <deque>

namespace jlm::hls
{

static rvsdg::Output *
follow_state_edge(
    rvsdg::Input * state_edge,
    std::vector<rvsdg::SimpleNode *> & mem_ops,
    bool modify);

static rvsdg::Output *
trace_edge(
    rvsdg::Input * state_edge,
    rvsdg::Output * new_edge,
    rvsdg::SimpleNode * target_call,
    rvsdg::Output * end)
{
  rvsdg::Input * previous_state_edge = nullptr;
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
    else if (auto ln = rvsdg::TryGetOwnerNode<LoopNode>(*state_edge))
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
      JLM_ASSERT(rvsdg::TryGetOwnerNode<LoopNode>(*out));
      JLM_ASSERT(out->region() == state_edge->region());
      state_edge = get_mem_state_user(out);
      continue;
    }

    auto si = state_edge;
    auto sn = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*si);
    auto new_si = new_edge_user;
    auto new_sn = &rvsdg::AssertGetOwnerNode<rvsdg::SimpleNode>(*new_si);
    auto [branchNode, branchOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<BranchOperation>(*state_edge);
    auto [muxNode, muxOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<MuxOperation>(*state_edge);
    if (branchOperation
        && !branchOperation
                ->loop) // this is an example of why preserving structural nodes would be nice
    {
      // start of gamma
      auto nbr = BranchOperation::create(*sn->input(0)->origin(), *new_edge);
      auto nmux = MuxOperation::create(*sn->input(0)->origin(), nbr, false)[0];
      new_edge_user->divert_to(nmux);
      rvsdg::Output * out = nullptr;
      for (size_t i = 0; i < sn->noutputs(); ++i)
      {
        out = trace_edge(get_mem_state_user(sn->output(i)), nbr[i], target_call, end);
      }
      JLM_ASSERT(out);
      auto [_, muxOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<MuxOperation>(*out);
      JLM_ASSERT(muxOperation);
      state_edge = get_mem_state_user(out);
      new_edge = nmux;
    }
    else if (branchOperation)
    {
      // end of loop
      auto [_, operation] = rvsdg::TryGetSimpleNodeAndOptionalOp<BranchOperation>(*new_edge_user);
      JLM_ASSERT(operation);
      JLM_ASSERT(branchOperation->loop);
      return util::AssertedCast<rvsdg::RegionResult>(get_mem_state_user(sn->output(0)))->output();
    }
    else if (muxOperation && !muxOperation->loop)
    {
      // end of gamma
      auto [_, operation] = rvsdg::TryGetSimpleNodeAndOptionalOp<MuxOperation>(*new_edge_user);
      JLM_ASSERT(operation);
      return sn->output(0);
    }
    else if (muxOperation)
    {
      // start of theta
      auto [_, operation] = rvsdg::TryGetSimpleNodeAndOptionalOp<MuxOperation>(*new_edge_user);
      JLM_ASSERT(operation);
      JLM_ASSERT(muxOperation->loop);
      state_edge = get_mem_state_user(sn->output(0));
      new_edge = new_sn->output(0);
    }
    else if (auto [node, op] =
                 rvsdg::TryGetSimpleNodeAndOptionalOp<LoopConstantBufferOperation>(*state_edge);
             op)
    {
      // start of loop
      auto [_, operation] = rvsdg::TryGetSimpleNodeAndOptionalOp<MuxOperation>(*new_edge_user);
      JLM_ASSERT(operation);
      state_edge = get_mem_state_user(sn->output(0));
      new_edge = new_sn->output(0);
    }
    else if (auto [node, op] =
                 rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::MemoryStateSplitOperation>(*state_edge);
             op)
    {
      rvsdg::Output * after_merge = nullptr;
      for (size_t i = 0; i < sn->noutputs(); ++i)
      {
        // pick right edge by searching for target
        std::vector<rvsdg::SimpleNode *> mem_ops;
        after_merge = follow_state_edge(get_mem_state_user(sn->output(i)), mem_ops, false);
        JLM_ASSERT(after_merge);
        auto [_, operation] =
            rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::MemoryStateMergeOperation>(*after_merge);
        JLM_ASSERT(operation);
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
        std::get<1>(
            rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::MemoryStateMergeOperation>(*state_edge))
        || std::get<1>(
            rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::LambdaExitMemoryStateMergeOperation>(
                *state_edge)))
    {
      // we did not split the new state
      return sn->output(0);
    }
    else if (auto [node, op] =
                 rvsdg::TryGetSimpleNodeAndOptionalOp<StateGateOperation>(*state_edge);
             op)
    {
      state_edge = get_mem_state_user(sn->output(si->index()));
    }
    else if (auto [node, op] =
                 rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::LoadNonVolatileOperation>(*state_edge);
             op)
    {
      state_edge = get_mem_state_user(sn->output(1));
    }
    else if (auto [node, op] =
                 rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::CallOperation>(*state_edge);
             op)
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

static void
handle_structural(
    std::vector<std::tuple<rvsdg::SimpleNode *, rvsdg::Input *>> & outstanding_dec_reqs,
    std::vector<rvsdg::SimpleNode *> & mem_ops,
    rvsdg::Input * state_edge_before,
    rvsdg::Output * state_edge_after)
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
    if (rvsdg::is<const llvm::StoreNonVolatileOperation>(op))
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
        std::vector<rvsdg::Output *> operands(
            { state_edge_after, split_outputs[1], split_outputs[2] });
        auto merge_out = llvm::MemoryStateMergeOperation::Create(operands);
        after_user->divert_to(merge_out);
        trace_edge(state_edge_req, split_outputs[1], req, state_edge_after);
        trace_edge(state_edge_req, split_outputs[2], resp, state_edge_after);
      }
    }
  }
}

static void
optimize_single_mem_op_loop(
    std::vector<rvsdg::SimpleNode *> & mem_ops,
    rvsdg::Input * state_edge_before,
    rvsdg::Output * state_edge_after)
{
  // the idea here is that if there is only one memory operation, and no other memory
  // operations/staet gates on a state edge in a loop we can remove the backedge part and treat the
  // edge like a loop constant, albeit with an output this will especially be important for stores
  // that have a response.
  // TODO: should this also be enabled for just memory state gates?
  if (mem_ops.size() == 1
      && (rvsdg::is<const llvm::StoreNonVolatileOperation>(mem_ops[0])
          || rvsdg::is<const llvm::LoadNonVolatileOperation>(mem_ops[0])))
  {
    // before and after belong to same loop node
    JLM_ASSERT(rvsdg::TryGetOwnerNode<LoopNode>(*state_edge_before));
    JLM_ASSERT(
        rvsdg::TryGetOwnerNode<LoopNode>(*state_edge_before)
        == rvsdg::TryGetOwnerNode<LoopNode>(*state_edge_after));
    convert_loop_state_to_lcb(state_edge_before);
  }
}

static rvsdg::Output *
follow_state_edge(
    rvsdg::Input * state_edge,
    std::vector<rvsdg::SimpleNode *> & mem_ops,
    bool modify)
{
  // we use input so we can handle the scenario of a store having an extra user to deq from addrq
  // things we can encounter:
  // * region result:
  //     * return associated output
  // * loop-node:
  //     * follow subregion
  //     * check which subset of dec_req, dec_resp and store is contained using handle_structural
  // * converted gamma:
  //     * need to follow all sub-paths
  //     * begins with branch with !br->loop
  //         * follow_state_edge
  //     * ends with mux
  //         * return mux output
  //     * check which subset of dec_req, dec_resp and store is contained using handle_structural
  // * mem state split/merge:
  //     * follow_state_edge/return merge output
  // * load/state-gate
  //     * continue on state output
  // * store/call
  //     * add to mem_ops
  //     * continue on state output
  //     * special case for store - can have multiple users because of addr_deq
  // this tracks decouple requests that have not been handled yet
  std::vector<std::tuple<rvsdg::SimpleNode *, rvsdg::Input *>> outstanding_dec_reqs;
  while (true)
  {
    if (auto rr = dynamic_cast<jlm::rvsdg::RegionResult *>(state_edge))
    {
      return rr->output();
    }
    else if (rvsdg::TryGetOwnerNode<LoopNode>(*state_edge))
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
    auto si = state_edge;
    auto sn = &rvsdg::AssertGetOwnerNode<rvsdg::SimpleNode>(*si);
    auto [branchNode, branchOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<BranchOperation>(*state_edge);
    auto [muxNode, muxOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<MuxOperation>(*state_edge);
    if (branchOperation
        && !branchOperation
                ->loop) // this is an example of why preserving structural nodes would be nice
    {
      std::vector<rvsdg::SimpleNode *> gamma_mem_ops;
      // start of gamma
      rvsdg::Output * out = nullptr;
      for (size_t i = 0; i < sn->noutputs(); ++i)
      {
        out = follow_state_edge(get_mem_state_user(sn->output(i)), gamma_mem_ops, modify);
      }
      auto [node, muxOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<MuxOperation>(*out);
      JLM_ASSERT(muxOperation);
      JLM_ASSERT(out);
      // get this here before the graph is modified by handle_structural
      auto new_state_edge = get_mem_state_user(out);
      if (modify)
        handle_structural(outstanding_dec_reqs, gamma_mem_ops, state_edge, out);
      state_edge = new_state_edge;
      mem_ops.insert(mem_ops.cend(), gamma_mem_ops.begin(), gamma_mem_ops.end());
    }
    else if (branchOperation)
    {
      // end of loop
      JLM_ASSERT(branchOperation->loop);
      return util::AssertedCast<rvsdg::RegionResult>(get_mem_state_user(sn->output(0)))->output();
    }
    else if (muxOperation && !muxOperation->loop)
    {
      // end of gamma
      return sn->output(0);
    }
    else if (muxOperation)
    {
      // start of theta
      JLM_ASSERT(muxOperation->loop);
      state_edge = get_mem_state_user(sn->output(0));
    }
    else if (auto [node, op] =
                 rvsdg::TryGetSimpleNodeAndOptionalOp<LoopConstantBufferOperation>(*state_edge);
             op)
    {
      // start of theta
      state_edge = get_mem_state_user(sn->output(0));
    }
    else if (
        std::get<1>(
            rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::MemoryStateSplitOperation>(*state_edge))
        || std::get<1>(
            rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::LambdaEntryMemoryStateSplitOperation>(
                *state_edge)))
    {
      for (size_t i = 0; i < sn->noutputs(); ++i)
      {
        auto followed = follow_state_edge(get_mem_state_user(sn->output(i)), mem_ops, modify);
        JLM_ASSERT(followed);
        JLM_ASSERT(
            std::get<1>(
                rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::MemoryStateMergeOperation>(*followed))
            || std::get<1>(
                rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::LambdaExitMemoryStateMergeOperation>(
                    *followed)));
        state_edge = get_mem_state_user(followed);
      }
    }
    else if (
        std::get<1>(
            rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::MemoryStateMergeOperation>(*state_edge))
        || std::get<1>(
            rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::LambdaExitMemoryStateMergeOperation>(
                *state_edge)))
    {
      return sn->output(0);
    }
    else if (std::get<1>(rvsdg::TryGetSimpleNodeAndOptionalOp<StateGateOperation>(*state_edge)))
    {
      mem_ops.push_back(sn);
      state_edge = get_mem_state_user(sn->output(si->index()));
    }
    else if (std::get<1>(
                 rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::LoadNonVolatileOperation>(*state_edge)))
    {
      mem_ops.push_back(sn);
      state_edge = get_mem_state_user(sn->output(1));
    }
    else if (std::get<1>(rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::CallOperation>(*state_edge)))
    {
      mem_ops.push_back(sn);
      state_edge = get_mem_state_user(sn->output(sn->noutputs() - 1));
    }
    else if (std::get<1>(rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::StoreNonVolatileOperation>(
                 *state_edge)))
    {
      mem_ops.push_back(sn);
      JLM_ASSERT(sn->output(0)->nusers() == 1);
      state_edge = get_mem_state_user(sn->output(0));
      // handle case of store that has one edge going off to deq
      if (std::get<1>(
              rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::MemoryStateSplitOperation>(*state_edge)))
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
convert_loop_state_to_lcb(rvsdg::Input * loop_state_input)
{
  JLM_ASSERT(rvsdg::is<rvsdg::StateType>(loop_state_input->Type()));
  JLM_ASSERT(rvsdg::TryGetOwnerNode<LoopNode>(*loop_state_input));
  auto si = util::AssertedCast<rvsdg::StructuralInput>(loop_state_input);
  auto arg = si->arguments.begin().ptr();
  auto user = get_mem_state_user(arg);
  auto [muxNode, muxOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<MuxOperation>(*user);
  JLM_ASSERT(muxOperation && muxOperation->loop);
  auto mux_node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*user);
  auto lcb_out = LoopConstantBufferOperation::create(
      *mux_node->input(0)->origin(),
      *mux_node->input(1)->origin())[0];
  mux_node->output(0)->divert_users(lcb_out);
  JLM_ASSERT(mux_node->IsDead());
  remove(mux_node);
}

void
decouple_mem_state(rvsdg::Region * region)
{
  JLM_ASSERT(region->nnodes() == 1);
  auto lambda = util::AssertedCast<const jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
  auto state_arg = &llvm::GetMemoryStateRegionArgument(*lambda);
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
  const auto entryNode = llvm::GetMemoryStateEntrySplit(*lambda);
  const auto exitNode = llvm::GetMemoryStateExitMerge(*lambda);
  JLM_ASSERT(entryNode->noutputs() == exitNode->ninputs());
  // process different pointer arg edges separately
  for (size_t i = 0; i < entryNode->noutputs(); ++i)
  {
    std::vector<rvsdg::SimpleNode *> mem_ops;
    std::vector<std::tuple<rvsdg::SimpleNode *, rvsdg::Input *>> dummy;
    follow_state_edge(get_mem_state_user(entryNode->output(i)), mem_ops, true);
    // we need this one final time across the whole lambda - at least for this edge
    handle_structural(
        dummy,
        mem_ops,
        get_mem_state_user(entryNode->output(i)),
        exitNode->input(i)->origin());
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

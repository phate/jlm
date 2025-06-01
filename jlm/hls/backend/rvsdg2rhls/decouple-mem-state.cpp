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

static rvsdg::output *
follow_state_edge(
    rvsdg::Input * state_edge,
    std::vector<rvsdg::SimpleNode *> & mem_ops,
    bool modify);

static rvsdg::output *
trace_edge(
    rvsdg::Input * state_edge,
    rvsdg::output * new_edge,
    rvsdg::SimpleNode * target_call,
    rvsdg::output * end)
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

    if (auto ln = rvsdg::TryGetOwnerNode<loop_node>(*state_edge))
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
      JLM_ASSERT(rvsdg::TryGetOwnerNode<loop_node>(*out));
      JLM_ASSERT(out->region() == state_edge->region());
      state_edge = get_mem_state_user(out);
    }
    else if (auto stateEdgeNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*state_edge))
    {
      auto newEdgeNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*new_edge_user);

      if (auto br = dynamic_cast<const branch_op *>(&stateEdgeNode->GetOperation()))
      {
        if (br->loop)
        {
          // end of loop
          JLM_ASSERT(rvsdg::is<branch_op>(newEdgeNode));
          return util::AssertedCast<rvsdg::RegionResult>(
                     get_mem_state_user(stateEdgeNode->output(0)))
              ->output();
        }

        // start of gamma
        auto nbr = branch_op::create(*stateEdgeNode->input(0)->origin(), *new_edge);
        auto nmux = mux_op::create(*stateEdgeNode->input(0)->origin(), nbr, false)[0];
        new_edge_user->divert_to(nmux);
        rvsdg::output * out = nullptr;
        for (size_t i = 0; i < stateEdgeNode->noutputs(); ++i)
        {
          out = trace_edge(get_mem_state_user(stateEdgeNode->output(i)), nbr[i], target_call, end);
        }
        auto outNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*out);
        JLM_ASSERT(rvsdg::is<mux_op>(outNode));
        state_edge = get_mem_state_user(out);
        new_edge = nmux;
      }
      else if (auto mux = dynamic_cast<const mux_op *>(&stateEdgeNode->GetOperation()))
      {
        if (!mux->loop)
        {
          // end of gamma
          JLM_ASSERT(rvsdg::is<mux_op>(newEdgeNode));
          return stateEdgeNode->output(0);
        }

        // start of theta
        JLM_ASSERT(rvsdg::is<mux_op>(newEdgeNode));
        JLM_ASSERT(mux->loop);
        state_edge = get_mem_state_user(stateEdgeNode->output(0));
        new_edge = newEdgeNode->output(0);
      }
      else if (rvsdg::is<loop_constant_buffer_op>(stateEdgeNode))
      {
        // start of loop
        JLM_ASSERT(rvsdg::is<mux_op>(newEdgeNode));
        state_edge = get_mem_state_user(stateEdgeNode->output(0));
        new_edge = newEdgeNode->output(0);
      }
      else if (rvsdg::is<llvm::MemoryStateSplitOperation>(stateEdgeNode))
      {
        rvsdg::output * after_merge = nullptr;
        for (size_t i = 0; i < stateEdgeNode->noutputs(); ++i)
        {
          // pick right edge by searching for target
          std::vector<rvsdg::SimpleNode *> mem_ops;
          after_merge =
              follow_state_edge(get_mem_state_user(stateEdgeNode->output(i)), mem_ops, false);
          auto afterMergeNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*after_merge);
          JLM_ASSERT(rvsdg::is<llvm::MemoryStateMergeOperation>(afterMergeNode));
          if (std::count(mem_ops.begin(), mem_ops.end(), target_call))
          {
            auto out = trace_edge(
                get_mem_state_user(stateEdgeNode->output(i)),
                new_edge,
                target_call,
                end);
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
          rvsdg::is<llvm::MemoryStateMergeOperation>(stateEdgeNode)
          || rvsdg::is<llvm::LambdaExitMemoryStateMergeOperation>(stateEdgeNode))
      {
        // we did not split the new state
        return stateEdgeNode->output(0);
      }
      else if (rvsdg::is<state_gate_op>(stateEdgeNode))
      {
        state_edge = get_mem_state_user(stateEdgeNode->output(state_edge->index()));
      }
      else if (rvsdg::is<llvm::LoadNonVolatileOperation>(stateEdgeNode))
      {
        state_edge = get_mem_state_user(stateEdgeNode->output(1));
      }
      else if (rvsdg::is<llvm::CallOperation>(stateEdgeNode))
      {
        auto state_origin = state_edge->origin();
        if (stateEdgeNode == target_call)
        {
          // move decouple call to new edge
          auto tmp = get_mem_state_user(stateEdgeNode->output(stateEdgeNode->noutputs() - 1));
          tmp->divert_to(state_origin);
          state_edge->divert_to(new_edge);
          state_edge = tmp;
          new_edge_user->divert_to(stateEdgeNode->output(stateEdgeNode->noutputs() - 1));
          new_edge = new_edge_user->origin();
        }
        else
        {
          state_edge = get_mem_state_user(stateEdgeNode->output(stateEdgeNode->noutputs() - 1));
        }
      }
      else
      {
        JLM_UNREACHABLE("whoops");
      }
    }
    else
    {
      JLM_UNREACHABLE("Unhandled node type.");
    }
  }
}

static void
handle_structural(
    std::vector<std::tuple<rvsdg::SimpleNode *, rvsdg::Input *>> & outstanding_dec_reqs,
    std::vector<rvsdg::SimpleNode *> & mem_ops,
    rvsdg::Input * state_edge_before,
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

static void
optimize_single_mem_op_loop(
    std::vector<rvsdg::SimpleNode *> & mem_ops,
    rvsdg::Input * state_edge_before,
    rvsdg::output * state_edge_after)
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
    JLM_ASSERT(rvsdg::TryGetOwnerNode<loop_node>(*state_edge_before));
    JLM_ASSERT(
        rvsdg::TryGetOwnerNode<loop_node>(*state_edge_before)
        == rvsdg::TryGetOwnerNode<loop_node>(*state_edge_after));
    convert_loop_state_to_lcb(state_edge_before);
  }
}

static rvsdg::output *
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

    if (rvsdg::TryGetOwnerNode<loop_node>(*state_edge))
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
    }
    else if (auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*state_edge))
    {
      if (auto br = dynamic_cast<const branch_op *>(&simpleNode->GetOperation()))
      {
        if (br->loop)
        {
          return util::AssertedCast<rvsdg::RegionResult>(get_mem_state_user(simpleNode->output(0)))
              ->output();
        }

        std::vector<rvsdg::SimpleNode *> gamma_mem_ops;
        // start of gamma
        rvsdg::output * out = nullptr;
        for (size_t i = 0; i < simpleNode->noutputs(); ++i)
        {
          out = follow_state_edge(get_mem_state_user(simpleNode->output(i)), gamma_mem_ops, modify);
        }
        auto outNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*out);
        JLM_ASSERT(rvsdg::is<mux_op>(outNode));
        // get this here before the graph is modified by handle_structural
        auto new_state_edge = get_mem_state_user(out);
        if (modify)
          handle_structural(outstanding_dec_reqs, gamma_mem_ops, state_edge, out);
        state_edge = new_state_edge;
        mem_ops.insert(mem_ops.cend(), gamma_mem_ops.begin(), gamma_mem_ops.end());
      }
      else if (auto mux = dynamic_cast<const mux_op *>(&simpleNode->GetOperation()))
      {
        if (!mux->loop)
        {
          // end of gamma
          return simpleNode->output(0);
        }

        // start of theta
        state_edge = get_mem_state_user(simpleNode->output(0));
      }
      else if (rvsdg::is<loop_constant_buffer_op>(simpleNode))
      {
        // start of theta
        state_edge = get_mem_state_user(simpleNode->output(0));
      }
      else if (
          rvsdg::is<llvm::MemoryStateSplitOperation>(simpleNode)
          || rvsdg::is<llvm::LambdaEntryMemoryStateSplitOperation>(simpleNode))
      {
        for (size_t i = 0; i < simpleNode->noutputs(); ++i)
        {
          auto followed =
              follow_state_edge(get_mem_state_user(simpleNode->output(i)), mem_ops, modify);
          JLM_ASSERT(followed);
          auto followedNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*followed);
          JLM_ASSERT(
              rvsdg::is<llvm::MemoryStateMergeOperation>(followedNode)
              || rvsdg::is<llvm::LambdaExitMemoryStateMergeOperation>(followedNode));
          state_edge = get_mem_state_user(followed);
        }
      }
      else if (
          rvsdg::is<llvm::MemoryStateMergeOperation>(simpleNode)
          || rvsdg::is<llvm::LambdaExitMemoryStateMergeOperation>(simpleNode))
      {
        return simpleNode->output(0);
      }
      else if (rvsdg::is<state_gate_op>(simpleNode))
      {
        mem_ops.push_back(simpleNode);
        state_edge = get_mem_state_user(simpleNode->output(state_edge->index()));
      }
      else if (rvsdg::is<llvm::LoadNonVolatileOperation>(simpleNode))
      {
        mem_ops.push_back(simpleNode);
        state_edge = get_mem_state_user(simpleNode->output(1));
      }
      else if (rvsdg::is<llvm::CallOperation>(simpleNode))
      {
        mem_ops.push_back(simpleNode);
        state_edge = get_mem_state_user(simpleNode->output(simpleNode->noutputs() - 1));
      }
      else if (rvsdg::is<llvm::StoreNonVolatileOperation>(simpleNode))
      {
        mem_ops.push_back(simpleNode);
        JLM_ASSERT(simpleNode->output(0)->nusers() == 1);
        state_edge = get_mem_state_user(simpleNode->output(0));
        // handle case of store that has one edge going off to deq
        auto [splitNode, splitOperation] =
            rvsdg::TryGetSimpleNodeAndOp<llvm::MemoryStateSplitOperation>(*state_edge);
        if (splitOperation)
        {
          // output 0 is the normal edge
          state_edge = get_mem_state_user(splitNode->output(0));
        }
      }
      else
      {
        JLM_UNREACHABLE("whoops");
      }
    }
    else
    {
      JLM_UNREACHABLE("Unhandled node type");
    }
  }
}

void
convert_loop_state_to_lcb(rvsdg::Input * loop_state_input)
{
  JLM_ASSERT(rvsdg::is<rvsdg::StateType>(loop_state_input->Type()));
  JLM_ASSERT(rvsdg::TryGetOwnerNode<loop_node>(*loop_state_input));
  auto si = util::AssertedCast<rvsdg::StructuralInput>(loop_state_input);
  auto arg = si->arguments.begin().ptr();
  auto user = get_mem_state_user(arg);
  auto [muxNode, muxOperation] = rvsdg::TryGetSimpleNodeAndOp<mux_op>(*user);
  JLM_ASSERT(muxOperation && muxOperation->loop);
  auto lcb_out = loop_constant_buffer_op::create(
      *muxNode->input(0)->origin(),
      *muxNode->input(1)->origin())[0];
  muxNode->output(0)->divert_users(lcb_out);
  JLM_ASSERT(muxNode->IsDead());
  remove(muxNode);
}

void
decouple_mem_state(rvsdg::Region * region)
{
  JLM_ASSERT(region->nnodes() == 1);
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
  auto [entryNode, entryOperation] =
      rvsdg::TryGetSimpleNodeAndOp<llvm::LambdaEntryMemoryStateSplitOperation>(*state_user);
  JLM_ASSERT(entryOperation);

  auto state_res = GetMemoryStateResult(*lambda);
  auto [exitNode, exitOperation] =
      rvsdg::TryGetSimpleNodeAndOp<llvm::LambdaExitMemoryStateMergeOperation>(*state_res->origin());
  JLM_ASSERT(exitOperation);
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

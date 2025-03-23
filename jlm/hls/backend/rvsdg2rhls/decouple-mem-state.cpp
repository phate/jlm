/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/decouple-mem-state.hpp>
#include <jlm/hls/backend/rvsdg2rhls/hls-function-util.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-conv.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-sep.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/view.hpp>

#include "jlm/hls/util/view.hpp"
#include "jlm/rvsdg/node.hpp"
#include "rhls-dne.hpp"
#include <deque>

namespace jlm::hls
{

bool
is_store(rvsdg::SimpleNode * node)
{
  return dynamic_cast<const llvm::StoreNonVolatileOperation *>(&node->GetOperation());
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
  std::cout << "trace_edge(" << std::hex << state_edge << ", " << std::hex << new_edge << ")"
            << std::endl;
  rvsdg::input * initial_state_edge = state_edge;
  rvsdg::input * previous_state_edge = nullptr;
  while (true)
  {

    if (previous_state_edge == state_edge)
      dump_dot(&state_edge->region()->graph()->GetRootRegion(), "trace_edge_dump.dot");
    // make sure we make progress
    JLM_ASSERT(previous_state_edge != state_edge);
    previous_state_edge = state_edge;
    // terminate once desired point is reached
    if (state_edge->origin() == end)
    {
      std::cout << "end reached" << std::endl;
      return end;
    }
    auto new_edge_user = get_mem_state_user(new_edge);
    std::cout << "state_edge = " << std::hex << state_edge << std::endl;
    std::cout << "new_edge_user = " << std::hex << new_edge_user << std::endl;
    JLM_ASSERT(new_edge_user->region() == state_edge->region());
    if (auto rr = dynamic_cast<jlm::rvsdg::RegionResult *>(state_edge))
    {
      return rr->output();
    }
    else if (auto ln = rvsdg::TryGetOwnerNode<loop_node>(*state_edge))
    {
      std::cout << "loop start" << std::endl;
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
      }
      state_edge = get_mem_state_user(out);
      continue;
    }
    auto si = util::AssertedCast<rvsdg::SimpleInput>(state_edge);
    auto sn = si->node();
    std::cout << "sn = " << std::hex << sn << std::endl;
    if (!dynamic_cast<rvsdg::SimpleInput *>(new_edge_user))
    {
      dump_dot(&state_edge->region()->graph()->GetRootRegion(), "trace_edge_dump.dot");
    }
    auto new_si = util::AssertedCast<rvsdg::SimpleInput>(new_edge_user);
    auto new_sn = new_si->node();
    std::cout << "new_sn = " << std::hex << new_sn << std::endl;
    auto br = TryGetOwnerOp<branch_op>(*state_edge);
    auto mux = TryGetOwnerOp<mux_op>(*state_edge);
    if (br && !br->loop) // this is an example of why preserving structural nodes would be nice
    {
      std::cout << "gamma start" << std::endl;
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
      if (!TryGetOwnerOp<mux_op>(*out))
      {
        std::unordered_map<rvsdg::output *, std::string> o_color;
        std::unordered_map<rvsdg::input *, std::string> i_color;
        i_color[initial_state_edge] = "orange";
        i_color[state_edge] = "red";
        o_color[out] = "green";
        o_color[new_edge] = "blue";
        dump_dot(
            &state_edge->region()->graph()->GetRootRegion(),
            "trace_edge_dump.dot",
            o_color,
            i_color);
      }
      //      if(out == end){
      //        return end;
      //      }
      JLM_ASSERT(TryGetOwnerOp<mux_op>(*out));
      state_edge = get_mem_state_user(out);
      new_edge = nmux;
    }
    else if (br)
    {
      std::cout << "loop end" << std::endl;
      // end of loop
      if (!TryGetOwnerOp<branch_op>(*new_edge_user))
        dump_dot(&state_edge->region()->graph()->GetRootRegion(), "trace_edge_dump.dot");
      JLM_ASSERT(TryGetOwnerOp<branch_op>(*new_edge_user));
      JLM_ASSERT(br->loop);
      return util::AssertedCast<rvsdg::RegionResult>(get_mem_state_user(sn->output(0)))->output();
    }
    else if (mux && !mux->loop)
    {
      std::cout << "gamma end" << std::endl;
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
    else if (TryGetOwnerOp<llvm::MemoryStateSplitOperation>(*state_edge))
    {
      std::cout << "mem split" << std::endl;
      rvsdg::output * after_merge = nullptr;
      bool follow_split = false;
      for (size_t i = 0; i < sn->noutputs(); ++i)
      {
        // pick right edge by searching for target
        std::vector<rvsdg::SimpleNode *> mem_ops;
        after_merge = follow_state_edge(state_edge, mem_ops, false);
        if (std::count(mem_ops.begin(), mem_ops.end(), target_call))
        {
          state_edge = get_mem_state_user(
              trace_edge(get_mem_state_user(sn->output(i)), new_edge, target_call, end));
          follow_split = true;
        }
      }
      if (!follow_split)
      {
        std::cout << "split skipped" << std::endl;
        // nothing relevant below the split - can just ignore it
        JLM_ASSERT(after_merge);
        state_edge = get_mem_state_user(after_merge);
      }
      new_edge = new_edge_user->origin();
    }
    else if (TryGetOwnerOp<llvm::MemoryStateMergeOperation>(*state_edge))
    {
      std::cout << "mem merge" << std::endl;
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
      dump_dot(&state_edge->region()->graph()->GetRootRegion(), "trace_edge_unreachable_dump.dot");
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
  JLM_ASSERT(state_edge_before->region()==state_edge_after->region());
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
        if (state_edge_after->region() != split_outputs[1]->region())
        {
          std::unordered_map<rvsdg::output *, std::string> o_color;
          std::unordered_map<rvsdg::input *, std::string> i_color;
          i_color[state_edge_req] = "orange";
          i_color[state_edge_before] = "green";
          o_color[state_edge_after] = "blue";
          o_color[split_outputs[1]] = "purple";
          dump_dot(
              &state_edge_after->region()->graph()->GetRootRegion(),
              "handle_structural_dump.dot",
              o_color,
              i_color);
        }
        JLM_ASSERT(state_edge_after->region() == split_outputs[1]->region());
        auto after_user = get_mem_state_user(state_edge_after);
        std::vector<rvsdg::output *> operands(
            { state_edge_after, split_outputs[1], split_outputs[2] });
        auto merge_out = llvm::MemoryStateMergeOperation::Create(operands);
        after_user->divert_to(merge_out);
        //        std::unordered_map<rvsdg::output *, std::string> o_color;
        //        std::unordered_map<rvsdg::input *, std::string> i_color;
        //        i_color[state_edge_req] = "orange";
        //        i_color[state_edge_before] = "green";
        //        o_color[state_edge_after] = "blue";
        //        dump_dot(&state_edge_after->region()->graph()->GetRootRegion(),
        //        "handle_structural_dump.dot", o_color, i_color);
        trace_edge(state_edge_req, split_outputs[1], req, state_edge_after);
        trace_edge(state_edge_req, split_outputs[2], resp, state_edge_after);
      }
    }
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
  rvsdg::input * state_edge_initial = state_edge;
  std::cout << "follow_state_edge(" << std::hex << state_edge << ")" << std::endl;
  // this tracks decouple requests that have not been handled yet
  std::vector<std::tuple<rvsdg::SimpleNode *, rvsdg::input *>> outstanding_dec_reqs;
  while (true)
  {
    std::cout << "state_edge = " << std::hex << state_edge << std::endl;
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
      if (!TryGetOwnerOp<mux_op>(*out))
      {
        std::unordered_map<rvsdg::output *, std::string> o_color;
        std::unordered_map<rvsdg::input *, std::string> i_color;
        i_color[state_edge_initial] = "orange";
        i_color[state_edge] = "green";
        o_color[out] = "blue";
        dump_dot(
            &state_edge->region()->graph()->GetRootRegion(),
            "follow_state_edge_dump.dot",
            o_color,
            i_color);
      }
      JLM_ASSERT(TryGetOwnerOp<mux_op>(*out));
      JLM_ASSERT(out);
      // get this here before the graph is modified by handle_structural
      auto new_state_edge = get_mem_state_user(out);
      if (modify)
        handle_structural(outstanding_dec_reqs, gamma_mem_ops, state_edge, out);
      state_edge = new_state_edge;
      mem_ops.insert(mem_ops.cend(), gamma_mem_ops.begin(), gamma_mem_ops.end());
      // TODO: are unnescessary state edge splits a problem? - and can a pass fix it?
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
    else if (
        TryGetOwnerOp<llvm::MemoryStateSplitOperation>(*state_edge)
        || TryGetOwnerOp<llvm::LambdaEntryMemoryStateSplitOperation>(*state_edge))
    {
      for (size_t i = 0; i < sn->noutputs(); ++i)
      {
        auto followed = follow_state_edge(get_mem_state_user(sn->output(i)), mem_ops, modify);
        JLM_ASSERT(followed);
//        if (!TryGetOwnerOp<llvm::MemoryStateMergeOperation>(*followed))
//        {
//          std::unordered_map<rvsdg::output *, std::string> o_color;
//          std::unordered_map<rvsdg::input *, std::string> i_color;
//          i_color[state_edge_initial] = "orange";
//          i_color[state_edge] = "green";
//          o_color[followed] = "blue";
//          dump_dot(
//              &state_edge->region()->graph()->GetRootRegion(),
//              "follow_state_edge_dump.dot",
//              o_color,
//              i_color);
//        }
        JLM_ASSERT(TryGetOwnerOp<llvm::MemoryStateMergeOperation>(*followed)||TryGetOwnerOp<llvm::LambdaExitMemoryStateMergeOperation>(*followed));
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
      state_edge = get_mem_state_user(sn->output(si->index()));
    }
    else if (TryGetOwnerOp<llvm::LoadNonVolatileOperation>(*state_edge))
    {
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
  // TODO: there could be a gamma around the loop

  std::vector<rvsdg::SimpleNode *> mem_ops;
  follow_state_edge(get_mem_state_user(state_arg), mem_ops, true);
  // TODO: check if mem-ops are sane
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

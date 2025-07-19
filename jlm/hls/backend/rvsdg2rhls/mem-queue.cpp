/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/decouple-mem-state.hpp>
#include <jlm/hls/backend/rvsdg2rhls/hls-function-util.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-conv.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-queue.hpp>
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
#include <deque>

void
jlm::hls::mem_queue(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = &graph.GetRootRegion();
  mem_queue(root);
}

void
find_load_store(
    jlm::rvsdg::Output * op,
    std::vector<jlm::rvsdg::SimpleNode *> & load_nodes,
    std::vector<jlm::rvsdg::SimpleNode *> & store_nodes,
    std::unordered_set<jlm::rvsdg::Output *> & visited)
{
  if (!jlm::rvsdg::is<jlm::llvm::MemoryStateType>(op->Type()))
  {
    return;
  }
  if (visited.count(op))
  {
    // skip already processed outputs
    return;
  }
  visited.insert(op);
  for (auto & user : op->Users())
  {
    if (auto simplenode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(user))
    {
      if (dynamic_cast<const jlm::llvm::StoreNonVolatileOperation *>(&simplenode->GetOperation()))
      {
        store_nodes.push_back(simplenode);
      }
      else if (dynamic_cast<const jlm::llvm::LoadNonVolatileOperation *>(
                   &simplenode->GetOperation()))
      {
        load_nodes.push_back(simplenode);
      }
      for (size_t i = 0; i < simplenode->noutputs(); ++i)
      {
        find_load_store(simplenode->output(i), load_nodes, store_nodes, visited);
      }
    }
    else if (auto sti = dynamic_cast<jlm::rvsdg::StructuralInput *>(&user))
    {
      for (auto & arg : sti->arguments)
      {
        find_load_store(&arg, load_nodes, store_nodes, visited);
      }
    }
    else if (auto r = dynamic_cast<jlm::rvsdg::RegionResult *>(&user))
    {
      if (auto ber = dynamic_cast<jlm::hls::backedge_result *>(r))
      {
        find_load_store(ber->argument(), load_nodes, store_nodes, visited);
      }
      else
      {
        find_load_store(r->output(), load_nodes, store_nodes, visited);
      }
    }
    else
    {
      JLM_UNREACHABLE("THIS SHOULD BE COVERED");
    }
  }
}

jlm::rvsdg::StructuralOutput *
find_loop_output(jlm::rvsdg::StructuralInput * sti)
{
  auto sti_arg = sti->arguments.first();
  JLM_ASSERT(sti_arg->nusers() == 1);
  auto & user = *sti_arg->Users().begin();
  auto [muxNode, muxOperation] = jlm::rvsdg::TryGetSimpleNodeAndOp<jlm::hls::MuxOperation>(user);
  JLM_ASSERT(muxNode && muxOperation);
  for (size_t i = 1; i < 3; ++i)
  {
    auto arg = muxNode->input(i)->origin();
    if (auto ba = dynamic_cast<jlm::hls::backedge_argument *>(arg))
    {
      auto res = ba->result();
      JLM_ASSERT(res);
      auto buffer_out = dynamic_cast<jlm::rvsdg::SimpleOutput *>(res->origin());
      JLM_ASSERT(buffer_out);
      JLM_ASSERT(jlm::rvsdg::is<jlm::hls::BufferOperation>(buffer_out->node()));
      auto branch_out =
          dynamic_cast<jlm::rvsdg::SimpleOutput *>(buffer_out->node()->input(0)->origin());
      JLM_ASSERT(branch_out);
      JLM_ASSERT(
          dynamic_cast<const jlm::hls::BranchOperation *>(&branch_out->node()->GetOperation()));
      // branch
      for (size_t j = 0; j < 2; ++j)
      {
        JLM_ASSERT(branch_out->node()->output(j)->nusers() == 1);
        auto result = dynamic_cast<jlm::rvsdg::RegionResult *>(
            &*branch_out->node()->output(j)->Users().begin());
        if (result)
        {
          return result->output();
        }
      }
    }
  }
  JLM_UNREACHABLE("This should never happen");
}

jlm::rvsdg::Output *
separate_load_edge(
    jlm::rvsdg::Output * mem_edge,
    jlm::rvsdg::Output * addr_edge,
    jlm::rvsdg::SimpleNode ** load,
    jlm::rvsdg::Output ** new_mem_edge,
    std::vector<jlm::rvsdg::Output *> & store_addresses,
    std::vector<jlm::rvsdg::Output *> & store_dequeues,
    std::vector<bool> & store_precedes,
    bool * load_encountered)
{
  // follows along mem edge and routes addr edge through the same regions
  // redirects the supplied load to the new edge and adds it to stores
  // the new edge might be routed through unnecessary regions. This should be fixed by running DNE
  while (true)
  {
    // each iteration should update common_edge and/or new_edge
    JLM_ASSERT(mem_edge->nusers() == 1);
    JLM_ASSERT(addr_edge->nusers() == 1);
    JLM_ASSERT(mem_edge != addr_edge);
    JLM_ASSERT(mem_edge->region() == addr_edge->region());
    auto user = &*mem_edge->Users().begin();
    auto & addr_edge_user = *addr_edge->Users().begin();
    if (dynamic_cast<jlm::rvsdg::RegionResult *>(user))
    {
      JLM_UNREACHABLE("THIS SHOULD NOT HAPPEN");
      // end of region reached
    }
    else if (auto sti = dynamic_cast<jlm::rvsdg::StructuralInput *>(user))
    {
      auto loop_node = jlm::util::AssertedCast<jlm::hls::loop_node>(sti->node());
      jlm::rvsdg::Output * buffer = nullptr;
      auto addr_edge_before_loop = addr_edge;
      addr_edge = loop_node->AddLoopVar(addr_edge, &buffer);
      addr_edge_user.divert_to(addr_edge);
      mem_edge = find_loop_output(sti);
      auto sti_arg = sti->arguments.first();
      JLM_ASSERT(sti_arg->nusers() == 1);
      auto & user = *sti_arg->Users().begin();
      auto [muxNode, muxOperation] =
          jlm::rvsdg::TryGetSimpleNodeAndOp<jlm::hls::MuxOperation>(user);
      JLM_ASSERT(muxNode && muxOperation);
      JLM_ASSERT(buffer->nusers() == 1);
      // use a separate vector to check if the loop contains stores
      std::vector<jlm::rvsdg::Output *> loop_store_addresses;
      separate_load_edge(
          muxNode->output(0),
          buffer,
          load,
          nullptr,
          loop_store_addresses,
          store_dequeues,
          store_precedes,
          load_encountered);
      if (loop_store_addresses.empty())
      {
        jlm::hls::convert_loop_state_to_lcb(&*addr_edge_before_loop->Users().begin());
      }
      else
      {
        store_addresses.insert(
            store_addresses.cend(),
            loop_store_addresses.begin(),
            loop_store_addresses.end());
      }
    }
    else if (auto sn = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*user))
    {
      auto op = &sn->GetOperation();

      if (auto br = dynamic_cast<const jlm::hls::BranchOperation *>(op))
      {
        if (!br->loop)
        {
          // start of gamma
          auto load_branch_out =
              jlm::hls::BranchOperation::create(*sn->input(0)->origin(), *addr_edge, false);
          for (size_t i = 0; i < sn->noutputs(); ++i)
          {
            // dummy user for edge
            auto dummy_user_tmp = jlm::hls::SinkOperation::create(*load_branch_out[i]);
            // Sink ops doesn't have any outputs so we get an empty vector back
            // But we are not allowed to discard the vector and can't have unused variables
            // So adding a meaningless assert to get it to compile
            JLM_ASSERT(dummy_user_tmp.size() == 0);
            auto dummy_user = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(
                *load_branch_out[i]->Users().begin());
            // need both load and common edge here
            load_branch_out[i] = separate_load_edge(
                sn->output(i),
                load_branch_out[i],
                load,
                &mem_edge,
                store_addresses,
                store_dequeues,
                store_precedes,
                load_encountered);
            JLM_ASSERT(load_branch_out[i]->nusers() == 1);
            JLM_ASSERT(dummy_user->input(0)->origin() == load_branch_out[i]);
            remove(dummy_user);
          }
          // create mux
          JLM_ASSERT(mem_edge->nusers() == 1);
          auto [muxNode, muxOperation] =
              jlm::rvsdg::TryGetSimpleNodeAndOp<jlm::hls::MuxOperation>(*mem_edge->Users().begin());
          JLM_ASSERT(muxNode && muxOperation);
          addr_edge = jlm::hls::MuxOperation::create(
              *muxNode->input(0)->origin(),
              load_branch_out,
              muxOperation->discarding,
              false)[0];
          addr_edge_user.divert_to(addr_edge);
          mem_edge = muxNode->output(0);
        }
        else
        {
          // end of loop
          auto [branchNode, branchOperation] =
              jlm::rvsdg::TryGetSimpleNodeAndOp<jlm::hls::BranchOperation>(addr_edge_user);
          JLM_ASSERT(branchNode && branchOperation);
          return nullptr;
        }
      }
      else if (auto mx = dynamic_cast<const jlm::hls::MuxOperation *>(op))
      {
        JLM_ASSERT(!mx->loop);
        // end of gamma
        JLM_ASSERT(new_mem_edge);
        *new_mem_edge = mem_edge;
        return addr_edge;
      }
      else if (dynamic_cast<const jlm::llvm::StoreNonVolatileOperation *>(op))
      {
        auto sg_out = jlm::hls::StateGateOperation::create(*sn->input(0)->origin(), { addr_edge });
        addr_edge = sg_out[1];
        addr_edge_user.divert_to(addr_edge);
        store_addresses.push_back(jlm::hls::route_to_region_rhls((*load)->region(), sg_out[0]));
        store_precedes.push_back(!*load_encountered);
        mem_edge = sn->output(0);
        JLM_ASSERT(mem_edge->nusers() == 1);
        user = &*mem_edge->Users().begin();
        auto [mssNode, msso] =
            jlm::rvsdg::TryGetSimpleNodeAndOp<jlm::llvm::MemoryStateSplitOperation>(*user);
        if (mssNode && msso)
        {
          // handle case where output of store is already connected to a MemStateSplit by adding an
          // output
          auto store_split =
              jlm::llvm::MemoryStateSplitOperation::Create(*mem_edge, msso->nresults() + 1);
          for (size_t i = 0; i < msso->nresults(); ++i)
          {
            mssNode->output(i)->divert_users(store_split[i]);
          }
          remove(mssNode);
          mem_edge = store_split[0];
          store_dequeues.push_back(
              jlm::hls::route_to_region_rhls((*load)->region(), store_split.back()));
        }
        else
        {
          auto store_split = jlm::llvm::MemoryStateSplitOperation::Create(*mem_edge, 2);
          mem_edge = store_split[0];
          user->divert_to(mem_edge);
          store_dequeues.push_back(
              jlm::hls::route_to_region_rhls((*load)->region(), store_split[1]));
        }
      }
      else if (auto lo = dynamic_cast<const jlm::llvm::LoadNonVolatileOperation *>(op))
      {
        JLM_ASSERT(sn->noutputs() == 2);
        if (sn == *load)
        {
          // create state gate for addr edge
          auto addr_sg_out =
              jlm::hls::StateGateOperation::create(*sn->input(0)->origin(), { addr_edge });
          addr_edge = addr_sg_out[1];
          addr_edge_user.divert_to(addr_edge);
          auto addr_sg_out2 = jlm::hls::StateGateOperation::create(*addr_sg_out[0], { addr_edge });
          addr_edge = addr_sg_out2[1];
          addr_edge_user.divert_to(addr_edge);
          // remove state edges from load
          auto new_load_outputs = jlm::llvm::LoadNonVolatileOperation::Create(
              addr_sg_out2[0],
              {},
              lo->GetLoadedType(),
              lo->GetAlignment());
          // create state gate for mem edge and load data
          auto mem_sg_out =
              jlm::hls::StateGateOperation::create(*new_load_outputs[0], { mem_edge });
          mem_edge = mem_sg_out[1];

          sn->output(0)->divert_users(new_load_outputs[0]);
          user->divert_to(addr_edge);
          sn->output(1)->divert_users(mem_edge);
          remove(sn);
          *load = dynamic_cast<jlm::rvsdg::SimpleOutput *>(new_load_outputs[0])->node();
          *load_encountered = true;
        }
        else
        {
          mem_edge = sn->output(1);
        }
      }
      else if (dynamic_cast<const jlm::hls::StateGateOperation *>(op))
      {
        mem_edge = sn->output(1);
      }
      else if (dynamic_cast<const jlm::llvm::CallOperation *>(op))
      {
        JLM_ASSERT("Decoupled nodes not implemented yet");
      }
      else if (dynamic_cast<const jlm::llvm::MemoryStateMergeOperation *>(op))
      {
        auto si_load_user = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(addr_edge_user);
        auto & userNode = jlm::rvsdg::AssertGetOwnerNode<jlm::rvsdg::SimpleNode>(*user);
        if (si_load_user && &userNode == sn)
        {
          return nullptr;
        }
        // TODO: handle
        JLM_UNREACHABLE("THIS SHOULD NOT HAPPEN");
      }
      else
      {
        JLM_UNREACHABLE("THIS SHOULD NOT HAPPEN");
      }
    }
    else
    {
      JLM_UNREACHABLE("THIS SHOULD NOT HAPPEN");
    }
  }
}

jlm::rvsdg::Output *
process_loops(jlm::rvsdg::Output * state_edge)
{
  while (true)
  {
    // each iteration should update state_edge
    JLM_ASSERT(state_edge->nusers() == 1);
    auto & user = *state_edge->Users().begin();
    if (dynamic_cast<jlm::rvsdg::RegionResult *>(&user))
    {
      // End of region reached
      return user.origin();
    }
    else if (auto sn = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(user))
    {
      auto op = &sn->GetOperation();
      auto br = dynamic_cast<const jlm::hls::BranchOperation *>(op);
      if (br && !br->loop)
      {
        // start of gamma
        for (size_t i = 0; i < sn->noutputs(); ++i)
        {
          state_edge = process_loops(sn->output(i));
        }
      }
      else if (jlm::rvsdg::is<jlm::hls::MuxOperation>(*op))
      {
        // end of gamma
        JLM_ASSERT(sn->noutputs() == 1);
        return sn->output(0);
      }
      else if (dynamic_cast<const jlm::llvm::LambdaExitMemoryStateMergeOperation *>(op))
      {
        // end of lambda
        JLM_ASSERT(sn->noutputs() == 1);
        return sn->output(0);
      }
      else if (dynamic_cast<const jlm::llvm::LoadNonVolatileOperation *>(op))
      {
        // load
        JLM_ASSERT(sn->noutputs() == 2);
        state_edge = sn->output(1);
      }
      else if (dynamic_cast<const jlm::llvm::CallOperation *>(op))
      {
        state_edge = sn->output(sn->noutputs() - 1);
      }
      else
      {
        JLM_ASSERT(sn->noutputs() == 1);
        state_edge = sn->output(0);
      }
    }
    else if (auto sti = dynamic_cast<jlm::rvsdg::StructuralInput *>(&user))
    {
      JLM_ASSERT(dynamic_cast<const jlm::hls::loop_node *>(sti->node()));
      // update to output of loop
      auto mem_edge_after_loop = find_loop_output(sti);
      JLM_ASSERT(mem_edge_after_loop->nusers() == 1);
      auto & common_user = *mem_edge_after_loop->Users().begin();

      std::vector<jlm::rvsdg::SimpleNode *> load_nodes;
      std::vector<jlm::rvsdg::SimpleNode *> store_nodes;
      std::unordered_set<jlm::rvsdg::Output *> visited;
      // this is a hack to keep search within the loop
      visited.insert(mem_edge_after_loop);
      find_load_store(&*sti->arguments.begin(), load_nodes, store_nodes, visited);
      auto split_states =
          jlm::llvm::MemoryStateSplitOperation::Create(*sti->origin(), load_nodes.size() + 1);
      // handle common edge
      auto mem_edge = split_states[0];
      sti->divert_to(mem_edge);
      split_states[0] = mem_edge_after_loop;
      state_edge = jlm::llvm::MemoryStateMergeOperation::Create(split_states);
      common_user.divert_to(state_edge);
      for (size_t i = 0; i < load_nodes.size(); ++i)
      {
        auto load = load_nodes[i];
        auto addr_edge = split_states[1 + i];
        std::vector<jlm::rvsdg::Output *> store_addresses;
        std::vector<jlm::rvsdg::Output *> store_dequeues;
        std::vector<bool> store_precedes;
        bool load_encountered = false;
        separate_load_edge(
            mem_edge,
            addr_edge,
            &load,
            nullptr,
            store_addresses,
            store_dequeues,
            store_precedes,
            &load_encountered);
        JLM_ASSERT(load_encountered);
        JLM_ASSERT(store_nodes.size() == store_addresses.size());
        JLM_ASSERT(store_nodes.size() == store_dequeues.size());
        auto state_gate_addr_in =
            dynamic_cast<jlm::rvsdg::SimpleOutput *>(load->input(0)->origin())->node()->input(0);
        for (size_t j = 0; j < store_nodes.size(); ++j)
        {
          JLM_ASSERT(state_gate_addr_in->origin()->region() == store_addresses[j]->region());
          JLM_ASSERT(store_dequeues[j]->region() == store_addresses[j]->region());
          state_gate_addr_in->divert_to(jlm::hls::AddressQueueOperation::create(
              *state_gate_addr_in->origin(),
              *store_addresses[j],
              *store_dequeues[j],
              store_precedes[j]));
        }
      }
    }
    else
    {
      JLM_UNREACHABLE("THIS SHOULD NOT HAPPEN");
    }
  }
}

void
jlm::hls::mem_queue(jlm::rvsdg::Region * region)
{
  auto lambda =
      jlm::util::AssertedCast<const jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
  auto state_arg = GetMemoryStateArgument(*lambda);
  if (!state_arg)
  {
    // No memstate, i.e., no memory used
    return;
  }
  // for each state edge:
  //    for each outer loop (theta/loop in lambda region):
  //        split state edge before the loop
  //         * one edge for only stores (preserves store order)
  //         * a separate edge for each load, going through the stores as well
  //        merge state edges after the loop
  //        for each load:
  //            insert store address queue before address input of load
  //             * enq order of stores guaranteed by load edge, deq by store edge
  //            for each store:
  //                insert state gate addr enq + deq after store complete

  // Check if there exists a memory state splitter
  if (state_arg->nusers() == 1)
  {
    auto entryNode = rvsdg::TryGetOwnerNode<rvsdg::Node>(*state_arg->Users().begin());
    if (jlm::rvsdg::is<const jlm::llvm::LambdaEntryMemoryStateSplitOperation>(
            entryNode->GetOperation()))
    {
      for (size_t i = 0; i < entryNode->noutputs(); ++i)
      {
        // Process each state edge separately
        jlm::rvsdg::Output * stateEdge = entryNode->output(i);
        process_loops(stateEdge);
      }
      return;
    }
  }
  // There is no memory state splitter, so process the single state edge in the graph
  process_loops(state_arg);
}

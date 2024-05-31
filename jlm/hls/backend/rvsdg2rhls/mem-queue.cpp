/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

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

#include <deque>

void
jlm::hls::mem_queue(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = graph.root();
  mem_queue(root);
}

void
dump_xml(const jlm::rvsdg::region * region, const std::string & file_name)
{
  auto xml_file = fopen(file_name.c_str(), "w");
  jlm::rvsdg::view_xml(region, xml_file);
  fclose(xml_file);
}

void
find_load_store(
    jlm::rvsdg::output * op,
    std::vector<jlm::rvsdg::simple_node *> & load_nodes,
    std::vector<jlm::rvsdg::simple_node *> & store_nodes,
    std::unordered_set<jlm::rvsdg::output *> & visited)
{
  if (!dynamic_cast<const jlm::llvm::MemoryStateType *>(&op->type()))
  {
    return;
  }
  if (visited.count(op))
  {
    // skip already processed outputs
    return;
  }
  visited.insert(op);
  for (auto user : *op)
  {
    if (auto si = dynamic_cast<jlm::rvsdg::simple_input *>(user))
    {
      auto simplenode = si->node();
      if (dynamic_cast<const jlm::llvm::StoreNonVolatileOperation *>(&simplenode->operation()))
      {
        store_nodes.push_back(simplenode);
      }
      else if (dynamic_cast<const jlm::llvm::LoadNonVolatileOperation *>(&simplenode->operation()))
      {
        load_nodes.push_back(simplenode);
      }
      for (size_t i = 0; i < simplenode->noutputs(); ++i)
      {
        find_load_store(simplenode->output(i), load_nodes, store_nodes, visited);
      }
    }
    else if (auto sti = dynamic_cast<jlm::rvsdg::structural_input *>(user))
    {
      for (auto & arg : sti->arguments)
      {
        find_load_store(&arg, load_nodes, store_nodes, visited);
      }
    }
    else if (auto r = dynamic_cast<jlm::rvsdg::result *>(user))
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

jlm::rvsdg::structural_output *
find_loop_output(jlm::rvsdg::structural_input * sti)
{
  auto sti_arg = sti->arguments.first();
  JLM_ASSERT(sti_arg->nusers() == 1);
  auto user = *sti_arg->begin();
  auto si = dynamic_cast<jlm::rvsdg::simple_input *>(user);
  JLM_ASSERT(dynamic_cast<const jlm::hls::mux_op *>(&si->node()->operation()));
  for (size_t i = 1; i < 3; ++i)
  {
    auto arg = si->node()->input(i)->origin();
    if (auto ba = dynamic_cast<jlm::hls::backedge_argument *>(arg))
    {
      auto res = ba->result();
      JLM_ASSERT(res);
      auto buffer_out = dynamic_cast<jlm::rvsdg::simple_output *>(res->origin());
      JLM_ASSERT(buffer_out);
      JLM_ASSERT(dynamic_cast<const jlm::hls::buffer_op *>(&buffer_out->node()->operation()));
      auto branch_out =
          dynamic_cast<jlm::rvsdg::simple_output *>(buffer_out->node()->input(0)->origin());
      JLM_ASSERT(branch_out);
      JLM_ASSERT(dynamic_cast<const jlm::hls::branch_op *>(&branch_out->node()->operation()));
      // branch
      for (size_t j = 0; j < 2; ++j)
      {
        JLM_ASSERT(branch_out->node()->output(j)->nusers() == 1);
        auto result = dynamic_cast<jlm::rvsdg::result *>(*branch_out->node()->output(j)->begin());
        if (result)
        {
          return result->output();
        }
      }
    }
  }
  JLM_UNREACHABLE("This should never happen");
}

std::deque<jlm::rvsdg::region *>
get_parent_regions(jlm::rvsdg::region * region)
{
  std::deque<jlm::rvsdg::region *> regions;
  jlm::rvsdg::region * target_region = region;
  while (!dynamic_cast<const jlm::llvm::lambda::operation *>(&target_region->node()->operation()))
  {
    regions.push_front(target_region);
    target_region = target_region->node()->region();
  }
  return regions;
}

jlm::rvsdg::output *
route_to_region(jlm::rvsdg::region * target, jlm::rvsdg::output * out)
{
  // create lists of nested regions
  std::deque<jlm::rvsdg::region *> target_regions = get_parent_regions(target);
  std::deque<jlm::rvsdg::region *> out_regions = get_parent_regions(out->region());
  JLM_ASSERT(target_regions.front() == out_regions.front());
  // remove common ancestor regions
  jlm::rvsdg::region * common_region = nullptr;
  while (!target_regions.empty() && !out_regions.empty()
         && target_regions.front() == out_regions.front())
  {
    common_region = target_regions.front();
    target_regions.pop_front();
    out_regions.pop_front();
  }
  JLM_ASSERT(common_region != nullptr);
  auto common_loop = dynamic_cast<jlm::hls::loop_node *>(common_region->node());
  JLM_ASSERT(common_loop);
  // route out to convergence point from out
  jlm::rvsdg::output * common_out = jlm::hls::route_request(common_region, out);
  // add a backedge to prevent cycles
  auto arg = common_loop->add_backedge(out->type());
  arg->result()->divert_to(common_out);
  // route inwards from convergence point to target
  auto result = jlm::hls::route_response(target, arg);
  return result;
}

jlm::rvsdg::output *
separate_load_edge(
    jlm::rvsdg::output * mem_edge,
    jlm::rvsdg::output * addr_edge,
    jlm::rvsdg::simple_node ** load,
    jlm::rvsdg::output ** new_mem_edge,
    std::vector<jlm::rvsdg::output *> & store_addresses,
    std::vector<jlm::rvsdg::output *> & store_dequeues,
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
    if (addr_edge->nusers() != 1)
    {
      dump_xml(addr_edge->region(), "no_users.rvsdg");
    }
    JLM_ASSERT(addr_edge->nusers() == 1);
    JLM_ASSERT(mem_edge != addr_edge);
    JLM_ASSERT(mem_edge->region() == addr_edge->region());
    auto user = *mem_edge->begin();
    auto addr_edge_user = *addr_edge->begin();
    if (dynamic_cast<jlm::rvsdg::result *>(user))
    {
      JLM_UNREACHABLE("THIS SHOULD NOT HAPPEN");
      // end of region reached
    }
    else if (auto sti = dynamic_cast<jlm::rvsdg::structural_input *>(user))
    {
      auto loop_node = dynamic_cast<jlm::hls::loop_node *>(sti->node());
      JLM_ASSERT(loop_node);
      jlm::rvsdg::output * buffer;

      addr_edge = loop_node->add_loopvar(addr_edge, &buffer);
      addr_edge_user->divert_to(addr_edge);
      mem_edge = find_loop_output(sti);
      auto sti_arg = sti->arguments.first();
      JLM_ASSERT(sti_arg->nusers() == 1);
      auto user = *sti_arg->begin();
      auto si = dynamic_cast<jlm::rvsdg::simple_input *>(user);
      JLM_ASSERT(dynamic_cast<const jlm::hls::mux_op *>(&si->node()->operation()));
      JLM_ASSERT(buffer->nusers() == 1);
      separate_load_edge(
          si->node()->output(0),
          buffer,
          load,
          nullptr,
          store_addresses,
          store_dequeues,
          store_precedes,
          load_encountered);
    }
    else if (auto si = dynamic_cast<jlm::rvsdg::simple_input *>(user))
    {
      auto sn = si->node();
      auto op = &si->node()->operation();

      if (auto br = dynamic_cast<const jlm::hls::branch_op *>(op))
      {
        if (!br->loop)
        {
          // start of gamma
          auto load_branch_out =
              jlm::hls::branch_op::create(*sn->input(0)->origin(), *addr_edge, false);
          for (size_t i = 0; i < sn->noutputs(); ++i)
          {
            // dummy user for edge
            auto dummy_user_tmp = jlm::hls::sink_op::create(*load_branch_out[i]);
            // Sink ops doesn't have any outputs so we get an empty vector back
            // But we are not allowed to discard the vector and can't have unused variables
            // So adding a meaningless assert to get it to compile
            JLM_ASSERT(dummy_user_tmp.size() == 0);
            auto dummy_user =
                dynamic_cast<jlm::rvsdg::simple_input *>(*load_branch_out[i]->begin())->node();
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
          auto mux_user = dynamic_cast<jlm::rvsdg::simple_input *>(*mem_edge->begin());
          JLM_ASSERT(mux_user);
          auto mux_op = dynamic_cast<const jlm::hls::mux_op *>(&mux_user->node()->operation());
          JLM_ASSERT(mux_op);
          addr_edge = jlm::hls::mux_op::create(
              *mux_user->node()->input(0)->origin(),
              load_branch_out,
              mux_op->discarding,
              false)[0];
          addr_edge_user->divert_to(addr_edge);
          mem_edge = mux_user->node()->output(0);
        }
        else
        {
          // end of loop
          auto load_user_input = dynamic_cast<jlm::rvsdg::simple_input *>(addr_edge_user);
          JLM_ASSERT(load_user_input);
          JLM_ASSERT(
              dynamic_cast<const jlm::hls::branch_op *>(&load_user_input->node()->operation()));
          return nullptr;
        }
      }
      else if (auto mx = dynamic_cast<const jlm::hls::mux_op *>(op))
      {
        JLM_ASSERT(!mx->loop);
        // end of gamma
        if (!new_mem_edge)
        {
          dump_xml(addr_edge->region(), "no_new_common_edge.rvsdg");
        }
        JLM_ASSERT(new_mem_edge);
        *new_mem_edge = mem_edge;
        return addr_edge;
      }
      else if (dynamic_cast<const jlm::llvm::StoreNonVolatileOperation *>(op))
      {
        auto sg_out = jlm::hls::state_gate_op::create(*sn->input(0)->origin(), { addr_edge });
        addr_edge = sg_out[1];
        addr_edge_user->divert_to(addr_edge);
        store_addresses.push_back(route_to_region((*load)->region(), sg_out[0]));
        store_precedes.push_back(!*load_encountered);
        mem_edge = sn->output(0);
        JLM_ASSERT(mem_edge->nusers() == 1);
        user = *mem_edge->begin();
        auto ui = dynamic_cast<jlm::rvsdg::simple_input *>(user);
        if (ui && dynamic_cast<const jlm::llvm::MemStateSplitOperator *>(&ui->node()->operation()))
        {
          auto msso =
              dynamic_cast<const jlm::llvm::MemStateSplitOperator *>(&ui->node()->operation());
          // handle case where output of store is already connected to a MemStateSplit by adding an
          // output
          auto store_split =
              jlm::llvm::MemStateSplitOperator::Create(mem_edge, msso->nresults() + 1);
          for (size_t i = 0; i < msso->nresults(); ++i)
          {
            ui->node()->output(i)->divert_users(store_split[i]);
          }
          remove(ui->node());
          mem_edge = store_split[0];
          store_dequeues.push_back(route_to_region((*load)->region(), store_split.back()));
        }
        else
        {
          auto store_split = jlm::llvm::MemStateSplitOperator::Create(mem_edge, 2);
          mem_edge = store_split[0];
          user->divert_to(mem_edge);
          store_dequeues.push_back(route_to_region((*load)->region(), store_split[1]));
        }
      }
      else if (auto lo = dynamic_cast<const jlm::llvm::LoadNonVolatileOperation *>(op))
      {
        JLM_ASSERT(sn->noutputs() == 2);
        if (sn == *load)
        {
          // create state gate for addr edge
          auto addr_sg_out =
              jlm::hls::state_gate_op::create(*sn->input(0)->origin(), { addr_edge });
          addr_edge = addr_sg_out[1];
          addr_edge_user->divert_to(addr_edge);
          auto addr_sg_out2 = jlm::hls::state_gate_op::create(*addr_sg_out[0], { addr_edge });
          addr_edge = addr_sg_out2[1];
          addr_edge_user->divert_to(addr_edge);
          // remove state edges from load
          auto new_load_outputs = jlm::llvm::LoadNonVolatileNode::Create(
              addr_sg_out2[0],
              {},
              lo->GetLoadedType(),
              lo->GetAlignment());
          // create state gate for mem edge and load data
          auto mem_sg_out = jlm::hls::state_gate_op::create(*new_load_outputs[0], { mem_edge });
          mem_edge = mem_sg_out[1];

          sn->output(0)->divert_users(mem_sg_out[0]);
          si->divert_to(addr_edge);
          sn->output(1)->divert_users(mem_edge);
          remove(sn);
          *load = dynamic_cast<jlm::rvsdg::simple_output *>(new_load_outputs[0])->node();
          *load_encountered = true;
        }
        else
        {
          mem_edge = sn->output(1);
        }
      }
      else if (dynamic_cast<const jlm::hls::state_gate_op *>(op))
      {
        mem_edge = sn->output(1);
      }
      else if (dynamic_cast<const jlm::llvm::CallOperation *>(op))
      {
        JLM_ASSERT("Decoupled nodes not implemented yet");
      }
      else if (dynamic_cast<const jlm::llvm::MemoryStateMergeOperation *>(op))
      {
        auto si_load_user = dynamic_cast<jlm::rvsdg::simple_input *>(addr_edge_user);
        if (si_load_user && si->node() == sn)
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

jlm::rvsdg::output *
process_loops(jlm::rvsdg::output * state_edge)
{
  while (true)
  {
    // each iteration should update state_edge
    JLM_ASSERT(state_edge->nusers() == 1);
    auto user = *state_edge->begin();
    if (dynamic_cast<jlm::rvsdg::result *>(user))
    {
      // end of region reached
      JLM_UNREACHABLE("This should never happen");
    }
    else if (auto si = dynamic_cast<jlm::rvsdg::simple_input *>(user))
    {
      auto sn = si->node();
      auto op = &si->node()->operation();
      auto br = dynamic_cast<const jlm::hls::branch_op *>(op);
      if (br && !br->loop)
      {
        // start of gamma
        for (size_t i = 0; i < sn->noutputs(); ++i)
        {
          state_edge = process_loops(sn->output(i));
        }
      }
      else if (dynamic_cast<const jlm::hls::mux_op *>(op))
      {
        // end of gamma
        JLM_ASSERT(sn->noutputs() == 1);
        return sn->output(0);
      }
      else if (dynamic_cast<const jlm::llvm::LambdaExitMemStateOperator *>(op))
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
      else
      {
        JLM_ASSERT(sn->noutputs() == 1);
        state_edge = sn->output(0);
      }
    }
    else if (auto sti = dynamic_cast<jlm::rvsdg::structural_input *>(user))
    {
      auto ln = dynamic_cast<jlm::hls::loop_node *>(sti->node());
      JLM_ASSERT(ln);
      // update to output of loop
      auto mem_edge_after_loop = find_loop_output(sti);
      JLM_ASSERT(mem_edge_after_loop->nusers() == 1);
      auto common_user = *mem_edge_after_loop->begin();

      std::vector<jlm::rvsdg::simple_node *> load_nodes;
      std::vector<jlm::rvsdg::simple_node *> store_nodes;
      std::unordered_set<jlm::rvsdg::output *> visited;
      // this is a hack to keep search within the loop
      visited.insert(mem_edge_after_loop);
      find_load_store(&*sti->arguments.begin(), load_nodes, store_nodes, visited);
      auto split_states =
          jlm::llvm::MemStateSplitOperator::Create(sti->origin(), load_nodes.size() + 1);
      // handle common edge
      auto mem_edge = split_states[0];
      sti->divert_to(mem_edge);
      split_states[0] = mem_edge_after_loop;
      state_edge = jlm::llvm::MemoryStateMergeOperation::Create(split_states);
      common_user->divert_to(state_edge);
      for (size_t i = 0; i < load_nodes.size(); ++i)
      {
        auto load = load_nodes[i];
        auto addr_edge = split_states[1 + i];
        std::vector<jlm::rvsdg::output *> store_addresses;
        std::vector<jlm::rvsdg::output *> store_dequeues;
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
            dynamic_cast<jlm::rvsdg::simple_output *>(load->input(0)->origin())->node()->input(0);
        for (size_t j = 0; j < store_nodes.size(); ++j)
        {
          JLM_ASSERT(state_gate_addr_in->origin()->region() == store_addresses[j]->region());
          JLM_ASSERT(store_dequeues[j]->region() == store_addresses[j]->region());
          state_gate_addr_in->divert_to(jlm::hls::addr_queue_op::create(
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
jlm::hls::mem_queue(jlm::rvsdg::region * region)
{
  auto lambda = dynamic_cast<const jlm::llvm::lambda::node *>(region->nodes.first());
  auto state_arg = GetMemoryStateArgument(*lambda);
  if (!state_arg)
  {
    // no memstate - i.e., no memory used
    return;
  }
  JLM_ASSERT(state_arg->nusers() == 1);
  auto state_user = *state_arg->begin();
  auto entry_input = dynamic_cast<jlm::rvsdg::simple_input *>(state_user);
  JLM_ASSERT(entry_input);
  auto entry_node = entry_input->node();
  JLM_ASSERT(
      dynamic_cast<const jlm::llvm::LambdaEntryMemStateOperator *>(&entry_node->operation()));
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

  for (size_t i = 0; i < entry_node->noutputs(); ++i)
  {
    jlm::rvsdg::output * state_edge = entry_node->output(i);
    process_loops(state_edge);
  }
}

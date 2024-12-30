//
// Created by david on 7/2/21.
//

#include <jlm/hls/backend/rvsdg2rhls/dae-conv.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rhls-dne.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/view.hpp>

#include <queue>

namespace jlm::hls
{

void
dae_conv(jlm::llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = &graph.GetRootRegion();
  dae_conv(root);
}

void
find_slice_node(rvsdg::Node * node, std::unordered_set<rvsdg::Node *> & slice);

void
find_slice_output(rvsdg::output * output, std::unordered_set<rvsdg::Node *> & slice)
{
  if (auto no = dynamic_cast<jlm::rvsdg::node_output *>(output))
  {
    if (slice.count(no->node()))
    {
      // only process each node once
      return;
    }
    slice.insert(no->node());
    JLM_ASSERT(slice.count(no->node()));
    find_slice_node(no->node(), slice);
  }
  else if (dynamic_cast<rvsdg::RegionArgument *>(output))
  {
    if (auto be = dynamic_cast<backedge_argument *>(output))
    {
      find_slice_output(be->result()->origin(), slice);
    }
  }
  else
  {
    JLM_UNREACHABLE("THIS SHOULDNT HAPPEN");
  }
}

void
find_slice_node(rvsdg::Node * node, std::unordered_set<rvsdg::Node *> & slice)
{
  for (size_t i = 0; i < node->ninputs(); ++i)
  {
    find_slice_output(node->input(i)->origin(), slice);
  }
}

void
find_data_slice_node(rvsdg::Node * node, std::unordered_set<rvsdg::Node *> & slice)
{
  for (size_t i = 0; i < node->ninputs(); ++i)
  {
    if (jlm::rvsdg::is<jlm::llvm::MemoryStateType>(node->input(i)->type()))
    {
      continue;
    }
    find_slice_output(node->input(i)->origin(), slice);
  }
}

void
find_state_slice_node(rvsdg::Node * node, std::unordered_set<rvsdg::Node *> & slice)
{
  for (size_t i = 0; i < node->ninputs(); ++i)
  {
    if (!jlm::rvsdg::is<jlm::llvm::MemoryStateType>(node->input(i)->type()))
    {
      continue;
    }
    find_slice_output(node->input(i)->origin(), slice);
  }
}

bool
is_slice_exclusive_(
    rvsdg::Node * source,
    rvsdg::Node * destination,
    std::unordered_set<jlm::rvsdg::Node *> & slice,
    std::unordered_set<jlm::rvsdg::Node *> & visited);

bool
is_slice_exclusive_input_(
    jlm::rvsdg::input * source,
    rvsdg::Node * destination,
    std::unordered_set<jlm::rvsdg::Node *> & slice,
    std::unordered_set<jlm::rvsdg::Node *> & visited)
{
  if (auto ni = dynamic_cast<jlm::rvsdg::node_input *>(source))
  {
    if (!is_slice_exclusive_(ni->node(), destination, slice, visited))
    {
      return false;
    }
  }
  else if (dynamic_cast<rvsdg::RegionResult *>(source))
  {
    if (auto be = dynamic_cast<backedge_result *>(source))
    {
      for (auto user : *be->argument())
      {
        if (!is_slice_exclusive_input_(user, destination, slice, visited))
        {
          return false;
        }
      }
    }
    else
    {
      // descendant escapes loop
      return false;
    }
  }
  else
  {
    JLM_UNREACHABLE("THIS SHOULDNT HAPPEN");
  }
  return true;
}

void
trace_to_loop_results(jlm::rvsdg::output * out, std::vector<rvsdg::RegionResult *> & results)
{
  for (auto user : *out)
  {
    if (auto res = dynamic_cast<rvsdg::RegionResult *>(user))
    {
      results.push_back(res);
    }
    else if (auto ni = dynamic_cast<jlm::rvsdg::node_input *>(user))
    {
      auto node = ni->node();
      for (size_t i = 0; i < node->noutputs(); ++i)
      {
        if (node->output(i)->type() == out->type())
        {
          trace_to_loop_results(node->output(i), results);
        }
      }
    }
    else
    {
      JLM_UNREACHABLE("NOPE");
    }
  }
}

bool
is_slice_exclusive_(
    rvsdg::Node * source,
    rvsdg::Node * destination,
    std::unordered_set<jlm::rvsdg::Node *> & slice,
    std::unordered_set<jlm::rvsdg::Node *> & visited)
{
  // check if descendents of source can leave the slice without going through destination
  if (source == destination)
  {
    return true;
  }
  else if (!slice.count(source))
  {
    return false;
  }
  else if (visited.count(source))
  {
    return true;
  }
  visited.insert(source);
  for (size_t i = 0; i < source->noutputs(); ++i)
  {
    for (auto user : *source->output(i))
    {
      if (!is_slice_exclusive_input_(user, destination, slice, visited))
      {
        return false;
      }
    }
  }
  return true;
}

bool
is_slice_exclusive_(
    rvsdg::Node * source,
    rvsdg::Node * destination,
    std::unordered_set<jlm::rvsdg::Node *> & slice)
{
  std::unordered_set<rvsdg::Node *> visited;
  return is_slice_exclusive_(source, destination, slice, visited);
}

void
dump_xml(const rvsdg::Region * region, const std::string & file_name)
{
  auto xml_file = fopen(file_name.c_str(), "w");
  jlm::rvsdg::view_xml(region, xml_file);
  fclose(xml_file);
}

void
decouple_load(
    loop_node * loopNode,
    jlm::rvsdg::SimpleNode * loadNode,
    std::unordered_set<rvsdg::Node *> & loop_slice)
{
  // loadNode is always a part of loop_slice due to state edges
  auto new_loop = loop_node::create(loopNode->region(), false);
  rvsdg::SubstitutionMap smap;
  std::vector<backedge_argument *> backedge_args;
  // create arguments
  for (size_t i = 0; i < loopNode->subregion()->narguments(); ++i)
  {
    auto arg = loopNode->subregion()->argument(i);
    // only process arguments that are used by at least one node in the slice
    for (auto user : *arg)
    {
      if (auto ni = dynamic_cast<jlm::rvsdg::node_input *>(user))
      {
        if (loop_slice.count(ni->node()))
        {
          rvsdg::RegionArgument * new_arg;
          if (auto be = dynamic_cast<backedge_argument *>(arg))
          {
            new_arg = new_loop->add_backedge(arg->Type());
            backedge_args.push_back(be);
          }
          else
          {
            auto new_in =
                rvsdg::StructuralInput::create(new_loop, arg->input()->origin(), arg->Type());
            smap.insert(arg->input(), new_in);
            new_arg = &EntryArgument::Create(*new_loop->subregion(), *new_in, arg->Type());
          }
          smap.insert(arg, new_arg);
          continue;
        }
      }
    }
  }
  // copy nodes
  std::vector<std::vector<rvsdg::Node *>> context(loopNode->subregion()->nnodes());
  for (auto & node : loopNode->subregion()->Nodes())
  {
    JLM_ASSERT(node.depth() < context.size());
    context[node.depth()].push_back(&node);
  }
  for (size_t n = 0; n < context.size(); n++)
  {
    for (auto node : context[n])
    {
      if (loop_slice.count(node))
      {
        node->copy(new_loop->subregion(), smap);
      }
    }
  }
  // handle backedges
  for (auto be : backedge_args)
  {
    auto res = be->result();
    auto new_res = dynamic_cast<backedge_argument *>(smap.lookup(be))->result();
    new_res->divert_to(smap.lookup(res->origin()));
  }

  // redirect state edges to new loop outputs
  for (size_t i = 1; i < loadNode->noutputs() - 1; ++i)
  {
    std::vector<rvsdg::RegionResult *> results;
    trace_to_loop_results(loadNode->output(i), results);
    JLM_ASSERT(results.size() <= 2);
    for (auto res : results)
    {
      if (dynamic_cast<backedge_result *>(res))
      {
        // backedges should already have been copied
        continue;
      }
      auto new_res_origin = smap.lookup(res->origin());
      auto new_state_output = rvsdg::StructuralOutput::create(new_loop, new_res_origin->Type());
      ExitResult::Create(*new_res_origin, *new_state_output);
      res->output()->divert_users(new_state_output);
    }
  }
  dump_xml(new_loop->subregion(), "new_loop_just_slice.rvsdg");
  // replace load with stategate
  auto new_load = dynamic_cast<jlm::rvsdg::node_output *>(smap.lookup(loadNode->output(0)))->node();
  std::vector<jlm::rvsdg::output *> in_states;
  std::vector<std::vector<jlm::rvsdg::input *>> state_users;
  for (size_t i = 1; i < new_load->ninputs() - 1; ++i)
  {
    in_states.push_back(new_load->input(i)->origin());
    state_users.emplace_back();
    for (auto user : *new_load->output(i))
    {
      state_users.back().push_back(user);
    }
  }

  auto gate_out = state_gate_op::create(*new_load->input(0)->origin(), in_states);
  // divert state edges to state gate
  for (size_t i = 0; i < in_states.size(); ++i)
  {
    for (auto user : state_users[i])
    {
      user->divert_to(gate_out[i + 1]);
    }
  }

  // create output for address
  auto load_addr = gate_out[0];
  auto addr_output = rvsdg::StructuralOutput::create(new_loop, load_addr->Type());
  ExitResult::Create(*load_addr, *addr_output);
  // trace and remove loop input for mem data reponse
  auto mem_data_loop_out = new_load->input(new_load->ninputs() - 1)->origin();
  auto mem_data_loop_arg = dynamic_cast<rvsdg::RegionArgument *>(mem_data_loop_out);
  auto mem_data_loop_in = mem_data_loop_arg->input();
  auto mem_data_resp = mem_data_loop_in->origin();
  dump_xml(new_loop->subregion(), "new_loop_before_remove.rvsdg");
  remove(new_load);
  JLM_ASSERT(mem_data_loop_arg->nusers() == 0);
  new_loop->subregion()->RemoveArgument(mem_data_loop_arg->index());
  new_loop->RemoveInput(mem_data_loop_in->index());
  // create new decoupled load
  auto dload_out = decoupled_load_op::create(*addr_output, *mem_data_resp);

  // redirect mem_req_addr to dload_out[1]
  auto old_mem_req_res =
      dynamic_cast<rvsdg::RegionResult *>(*loadNode->output(loadNode->noutputs() - 1)->begin());
  auto old_mem_req_out = old_mem_req_res->output();
  auto mem_req_in = *old_mem_req_out->begin();
  mem_req_in->divert_to(dload_out[1]);
  loopNode->subregion()->RemoveResult(old_mem_req_res->index());
  loopNode->RemoveOutput(old_mem_req_out->index());
  // redirect state outputs of loadNode so they become pass through
  for (size_t i = 1; i < loadNode->noutputs() - 1; ++i)
  {
    loadNode->output(i)->divert_users(loadNode->input(i)->origin());
  }
  // use a buffer here to make ready logic for response easy and consistent
  auto buf = buffer_op::create(*dload_out[0], 2, true)[0];
  // replace data output of loadNode
  auto old_data_in = rvsdg::StructuralInput::create(loopNode, buf, dload_out[0]->Type());
  auto & old_data_arg =
      EntryArgument::Create(*loopNode->subregion(), *old_data_in, dload_out[0]->Type());
  loadNode->output(0)->divert_users(&old_data_arg);
  remove(loadNode);
}

bool
process_loopnode(loop_node * loopNode)
{
  for (auto & node : jlm::rvsdg::topdown_traverser(loopNode->subregion()))
  {
    if (auto ln = dynamic_cast<loop_node *>(node))
    {
      if (process_loopnode(ln))
      {
        return true;
      }
    }
    else if (auto simplenode = dynamic_cast<jlm::rvsdg::SimpleNode *>(node))
    {
      if (dynamic_cast<const load_op *>(&simplenode->GetOperation()))
      {
        // can currently only generate dae one loop deep
        // find load slice within loop - three slices - complete, data and state-edge
        std::unordered_set<rvsdg::Node *> loop_slice, data_slice, state_slice;
        find_slice_node(simplenode, loop_slice);
        find_data_slice_node(simplenode, data_slice);
        find_state_slice_node(simplenode, state_slice);
        // check if this load can be decoupled
        bool can_decouple = true;
        for (auto sn : data_slice)
        {
          if (rvsdg::is<rvsdg::StructuralOperation>(sn))
          {
            // data slice may not contain loops
            can_decouple = false;
            break;
          }
          else if (
              dynamic_cast<const load_op *>(&sn->GetOperation())
              || dynamic_cast<const store_op *>(&sn->GetOperation()))
          {
            // data slice may not contain loads or stores - this includes node
            can_decouple = false;
            break;
          }
          else if (dynamic_cast<const decoupled_load_op *>(&sn->GetOperation()))
          {
            // decoupled load has to be exclusive to load slice - e.g. not needed once load slice is
            // removed
            if (!is_slice_exclusive_(sn, simplenode, loop_slice))
            {
              can_decouple = false;
              break;
            }
          }
        }
        JLM_ASSERT(!can_decouple || !data_slice.count(simplenode));
        for (auto sn : state_slice)
        {
          if (rvsdg::is<rvsdg::StructuralOperation>(sn))
          {
            // state slice may not contain loops
            can_decouple = false;
            break;
          }
          else if (
              dynamic_cast<const load_op *>(&sn->GetOperation())
              || dynamic_cast<const store_op *>(&sn->GetOperation()))
          {
            // state slice may not contain loads or stores except for node
            if (sn != dynamic_cast<rvsdg::Node *>(simplenode))
            {
              can_decouple = false;
              break;
            }
          }
          else if (dynamic_cast<const decoupled_load_op *>(&sn->GetOperation()))
          {
            // decoupled load has to be exclusive to load slice - e.g. not needed once load slice is
            // removed
            if (!is_slice_exclusive_(sn, simplenode, loop_slice))
            {
              can_decouple = false;
              break;
            }
          }
        }
        if (can_decouple)
        {
          decouple_load(loopNode, simplenode, loop_slice);
          //                    dump_xml(loopNode->subregion(), "decoupled_before_dne.rvsdg");
          //                    dne(loopNode->subregion());
          //                    remove_unused_loop_inputs(loopNode);
          //                    dump_xml(loopNode->subregion(), "decoupled_after_dne.rvsdg");
          return true;
        }
      }
    }
  }
  return false;
}

void
dae_conv(rvsdg::Region * region)
{
  auto lambda = dynamic_cast<const jlm::llvm::lambda::node *>(region->Nodes().begin().ptr());
  bool changed;
  do
  {
    changed = false;
    for (auto & node : jlm::rvsdg::topdown_traverser(lambda->subregion()))
    {
      if (auto loopnode = dynamic_cast<loop_node *>(node))
      {
        if (process_loopnode(loopnode))
        {
          changed = true;
          break;
        }
      }
    }
    if (changed)
    {
      // run dne after each change
      dne(lambda->subregion());
    }
  } while (changed);
}

} // namespace jlm::hls

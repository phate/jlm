/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/alloca-conv.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-conv.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

class TraceAllocaUses
{
public:
  std::vector<jlm::rvsdg::simple_node *> load_nodes;
  std::vector<jlm::rvsdg::simple_node *> store_nodes;

  TraceAllocaUses(jlm::rvsdg::output * op)
  {
    trace(op);
  }

private:
  void
  trace(jlm::rvsdg::output * op)
  {
    if (!dynamic_cast<const jlm::llvm::PointerType *>(&op->type()))
    {
      // only process pointer outputs
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
        else if (dynamic_cast<const jlm::llvm::LoadNonVolatileOperation *>(
                     &simplenode->operation()))
        {
          load_nodes.push_back(simplenode);
        }
        else if (dynamic_cast<const jlm::llvm::CallOperation *>(&simplenode->operation()))
        {
          // TODO: verify this is the right type of function call
          throw jlm::util::error("encountered a call for an alloca");
        }
        else
        {
          for (size_t i = 0; i < simplenode->noutputs(); ++i)
          {
            trace(simplenode->output(i));
          }
        }
      }
      else if (auto sti = dynamic_cast<jlm::rvsdg::structural_input *>(user))
      {
        for (auto & arg : sti->arguments)
        {
          trace(&arg);
        }
      }
      else if (auto r = dynamic_cast<jlm::rvsdg::result *>(user))
      {
        if (auto ber = dynamic_cast<backedge_result *>(r))
        {
          trace(ber->argument());
        }
        else
        {
          trace(r->output());
        }
      }
      else
      {
        JLM_UNREACHABLE("THIS SHOULD BE COVERED");
      }
    }
  }

  std::unordered_set<jlm::rvsdg::output *> visited;
};

jlm::rvsdg::output *
gep_to_index(jlm::rvsdg::output * o)
{
  // TODO: handle geps that are not direct predecessors
  auto no = dynamic_cast<jlm::rvsdg::node_output *>(o);
  JLM_ASSERT(no);
  auto gep = dynamic_cast<const jlm::llvm::GetElementPtrOperation *>(&no->node()->operation());
  JLM_ASSERT(gep);
  // pointer to array, i.e. first index is zero
  // TODO: check
  JLM_ASSERT(no->node()->ninputs() == 3);
  return no->node()->input(2)->origin();
}

void
alloca_conv(jlm::rvsdg::region * region)
{
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto structnode = dynamic_cast<jlm::rvsdg::structural_node *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
      {
        alloca_conv(structnode->subregion(n));
      }
    }
    else if (auto po = dynamic_cast<const jlm::llvm::alloca_op *>(&(node->operation())))
    {
      // ensure that the size is one
      JLM_ASSERT(node->ninputs() == 1);
      auto constant_output = dynamic_cast<jlm::rvsdg::node_output *>(node->input(0)->origin());
      JLM_ASSERT(constant_output);
      auto constant_operation =
          dynamic_cast<const jlm::rvsdg::bitconstant_op *>(&constant_output->node()->operation());
      JLM_ASSERT(constant_operation);
      JLM_ASSERT(constant_operation->value().to_uint() == 1);
      // ensure that the alloca is an array type
      auto at = std::dynamic_pointer_cast<const jlm::llvm::arraytype>(po->ValueType());
      JLM_ASSERT(at);
      // detect loads and stores attached to alloca
      TraceAllocaUses ta(node->output(0));
      // create memory + response
      auto mem_outs = local_mem_op::create(at, node->region());
      auto resp_outs = local_mem_resp_op::create(*mem_outs[0], ta.load_nodes.size());
      std::cout << "alloca converted " << at->debug_string() << std::endl;
      // replace gep outputs (convert pointer to index calculation)
      // replace loads and stores
      std::vector<jlm::rvsdg::output *> load_addrs;
      for (auto l : ta.load_nodes)
      {
        auto index = gep_to_index(l->input(0)->origin());
        auto response = route_response(l->region(), resp_outs.front());
        resp_outs.erase(resp_outs.begin());
        std::vector<jlm::rvsdg::output *> states;
        for (size_t i = 1; i < l->ninputs(); ++i)
        {
          states.push_back(l->input(i)->origin());
        }
        auto load_outs = local_load_op::create(*index, states, *response);
        auto nn = dynamic_cast<jlm::rvsdg::node_output *>(load_outs[0])->node();
        for (size_t i = 0; i < l->noutputs(); ++i)
        {
          l->output(i)->divert_users(nn->output(i));
        }
        remove(l);
        auto addr = route_request(node->region(), load_outs.back());
        load_addrs.push_back(addr);
      }
      std::vector<jlm::rvsdg::output *> store_operands;
      for (auto s : ta.store_nodes)
      {
        auto index = gep_to_index(s->input(0)->origin());
        std::vector<jlm::rvsdg::output *> states;
        for (size_t i = 2; i < s->ninputs(); ++i)
        {
          states.push_back(s->input(i)->origin());
        }
        auto store_outs = local_store_op::create(*index, *s->input(1)->origin(), states);
        auto nn = dynamic_cast<jlm::rvsdg::node_output *>(store_outs[0])->node();
        for (size_t i = 0; i < s->noutputs(); ++i)
        {
          s->output(i)->divert_users(nn->output(i));
        }
        remove(s);
        auto addr = route_request(node->region(), store_outs[store_outs.size() - 2]);
        auto data = route_request(node->region(), store_outs.back());
        store_operands.push_back(addr);
        store_operands.push_back(data);
      }
      // TODO: ensure that loads/stores are either alloca or global, never both
      // TODO: ensure that loads/stores have same width and alignment and geps can be merged -
      // otherwise slice? create request
      auto req_outs = local_mem_req_op::create(*mem_outs[1], load_addrs, store_operands);

      // remove alloca from memstate merge
      // TODO: handle general case of other nodes getting state edge without a merge
      JLM_ASSERT(node->output(1)->nusers() == 1);
      auto merge_in = *node->output(1)->begin();
      auto merge_node = rvsdg::input::GetNode(*merge_in);
      if (dynamic_cast<const llvm::MemoryStateMergeOperation *>(&merge_node->operation()))
      {
        // merge after alloca -> remove merge
        JLM_ASSERT(merge_node->ninputs() == 2);
        auto other_index = merge_in->index() ? 0 : 1;
        merge_node->output(0)->divert_users(merge_node->input(other_index)->origin());
        jlm::rvsdg::remove(merge_node);
      }
      else
      {
        // TODO: fix this properly by adding a state edge to the LambdaEntryMemState and routing it
        // to the region
        JLM_ASSERT(false);
      }

      // TODO: run dne to
      //  remove loads/stores
      //  remove geps
      //  remove alloca pointer users
      //  remove alloca
    }
  }
}

void
alloca_conv(jlm::llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = graph.root();
  alloca_conv(root);
}

} // namespace jlm::hls

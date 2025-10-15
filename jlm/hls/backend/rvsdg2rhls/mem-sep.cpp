/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/add-prints.hpp>
#include <jlm/hls/backend/rvsdg2rhls/hls-function-util.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-conv.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-sep.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/view.hpp>

#include <algorithm>

namespace jlm::hls
{

rvsdg::RegionResult *
trace_edge(
    jlm::rvsdg::Output * common_edge,
    jlm::rvsdg::Output * new_edge,
    std::vector<rvsdg::Node *> & load_nodes,
    const std::vector<rvsdg::Node *> & store_nodes,
    std::vector<rvsdg::Node *> & decouple_nodes)
{
  // follows along common edge and routes new edge through the same regions
  // redirects the supplied loads, stores and decouples to the new edge
  // the new edge might be routed through unnecessary regions. This should be fixed by running DNE
  while (true)
  {
    // each iteration should update common_edge and/or new_edge
    JLM_ASSERT(common_edge->nusers() == 1);
    JLM_ASSERT(new_edge->nusers() == 1);
    auto & user = *common_edge->Users().begin();
    auto & new_next = *new_edge->Users().begin();
    if (auto res = dynamic_cast<rvsdg::RegionResult *>(&user))
    {
      // end of region reached
      return res;
    }
    else if (auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(user))
    {
      auto ip = gammaNode->AddEntryVar(new_edge);
      std::vector<jlm::rvsdg::Output *> vec;
      new_edge = gammaNode->AddExitVar(ip.branchArgument).output;
      new_next.divert_to(new_edge);

      auto rolevar = gammaNode->MapInput(user);

      if (auto entryvar = std::get_if<rvsdg::GammaNode::EntryVar>(&rolevar))
      {
        for (size_t i = 0; i < gammaNode->nsubregions(); ++i)
        {
          auto subres = trace_edge(
              entryvar->branchArgument[i],
              ip.branchArgument[i],
              load_nodes,
              store_nodes,
              decouple_nodes);
          common_edge = subres->output();
        }
      }
    }
    else if (auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(user))
    {
      auto olv = theta->MapInputLoopVar(user);
      auto lv = theta->AddLoopVar(new_edge);
      trace_edge(olv.pre, lv.pre, load_nodes, store_nodes, decouple_nodes);
      common_edge = olv.output;
      new_edge = lv.output;
      new_next.divert_to(new_edge);
    }
    else if (auto sn = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(user))
    {
      auto op = &sn->GetOperation();
      if (dynamic_cast<const jlm::llvm::StoreNonVolatileOperation *>(op))
      {
        JLM_ASSERT(sn->noutputs() == 1);
        if (store_nodes.end() != std::find(store_nodes.begin(), store_nodes.end(), sn))
        {
          user.divert_to(new_edge);
          sn->output(0)->divert_users(common_edge);
          new_edge = sn->output(0);
          new_next.divert_to(new_edge);
        }
        else
        {
          common_edge = sn->output(0);
        }
      }
      else if (dynamic_cast<const jlm::llvm::LoadNonVolatileOperation *>(op))
      {
        JLM_ASSERT(sn->noutputs() == 2);
        if (load_nodes.end() != std::find(load_nodes.begin(), load_nodes.end(), sn))
        {
          auto & new_next = *new_edge->Users().begin();
          user.divert_to(new_edge);
          sn->output(1)->divert_users(common_edge);
          new_next.divert_to(sn->output(1));
          new_edge = sn->output(1);
          new_next.divert_to(new_edge);
        }
        else
        {
          common_edge = sn->output(1);
        }
      }
      else if (dynamic_cast<const jlm::llvm::CallOperation *>(op))
      {
        int oi = sn->noutputs() - sn->ninputs() + user.index();
        // TODO: verify this is the right type of function call
        if (decouple_nodes.end() != std::find(decouple_nodes.begin(), decouple_nodes.end(), sn))
        {
          auto & new_next = *new_edge->Users().begin();
          user.divert_to(new_edge);
          sn->output(oi)->divert_users(common_edge);
          new_next.divert_to(sn->output(oi));
          new_edge = new_next.origin();
        }
        else
        {
          common_edge = sn->output(oi);
        }
      }
      else
      {
        JLM_ASSERT(sn->noutputs() == 1);
        common_edge = sn->output(0);
      }
    }
    else
    {
      JLM_UNREACHABLE("THIS SHOULD NOT HAPPEN");
    }
  }
}

void
gather_other_calls(rvsdg::Region * region, std::vector<jlm::rvsdg::SimpleNode *> & calls)
{
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
        gather_other_calls(structnode->subregion(n), calls);
    }
    else if (auto simplenode = dynamic_cast<jlm::rvsdg::SimpleNode *>(node))
    {
      if (dynamic_cast<const llvm::CallOperation *>(&simplenode->GetOperation()))
      {
        auto name = jlm::hls::get_function_name(simplenode->input(0));
        // only non-decouple callse
        if (name.rfind("decouple") == name.npos)
          calls.push_back(simplenode);
      }
    }
  }
}

/* assign each pointer argument its own state edge. */
void
mem_sep_argument(rvsdg::Region * region)
{
  auto lambda = dynamic_cast<const rvsdg::LambdaNode *>(region->Nodes().begin().ptr());
  auto lambda_region = lambda->subregion();
  auto state_arg = &llvm::GetMemoryStateRegionArgument(*lambda);
  if (!state_arg)
  {
    // no memstate - i.e., no memory used
    return;
  }

  auto & state_user = *state_arg->Users().begin();
  auto tracedPointerNodesVector = TracePointerArguments(lambda);
  for (auto & tp : tracedPointerNodesVector)
  {
    auto & decouple_nodes = tp.decoupleNodes;
    auto decouple_requests_cnt = decouple_nodes.size();
    // place decouple responses along same state edge
    for (size_t i = 0; i < decouple_requests_cnt; ++i)
    {
      auto req = decouple_nodes[i];
      auto channel = req->input(1)->origin();
      auto channel_constant = jlm::hls::trace_constant(channel);
      auto decouple_response = jlm::hls::find_decouple_response(lambda, channel_constant);
      decouple_nodes.push_back(decouple_response);
    }
  }
  // create fake ports for non-decouple calls
  std::vector<jlm::rvsdg::SimpleNode *> other_calls;
  gather_other_calls(lambda_region, other_calls);
  for (auto call : other_calls)
  {
    tracedPointerNodesVector.emplace_back();
    tracedPointerNodesVector.back().decoupleNodes.push_back(call);
  }
  auto entry_states = jlm::llvm::LambdaEntryMemoryStateSplitOperation::Create(
      *state_arg,
      1 + tracedPointerNodesVector.size());
  auto state_result = &llvm::GetMemoryStateRegionResult(*lambda);
  // handle existing state edge - TODO: remove entirely?
  auto common_edge = entry_states.back();
  entry_states.pop_back();
  state_user.divert_to(common_edge);
  entry_states.push_back(state_result->origin());
  auto & merged_state =
      jlm::llvm::LambdaExitMemoryStateMergeOperation::Create(*lambda_region, entry_states);
  entry_states.pop_back();
  state_result->divert_to(&merged_state);

  for (auto tp : tracedPointerNodesVector)
  {
    auto new_edge = entry_states.back();
    entry_states.pop_back();
    trace_edge(common_edge, new_edge, tp.loadNodes, tp.storeNodes, tp.decoupleNodes);
  }
}

MemoryStateSeparation::~MemoryStateSeparation() noexcept = default;

MemoryStateSeparation::MemoryStateSeparation()
    : Transformation("MemoryStateSeparation")
{}

void
MemoryStateSeparation::Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector &)
{
  const auto & graph = rvsdgModule.Rvsdg();
  const auto rootRegion = &graph.GetRootRegion();
  if (rootRegion->numNodes() != 1)
  {
    throw std::logic_error("Root should have only one node now");
  }

  const auto lambdaNode =
      dynamic_cast<const rvsdg::LambdaNode *>(rootRegion->Nodes().begin().ptr());
  if (!lambdaNode)
  {
    throw std::logic_error("Node needs to be a lambda");
  }

  mem_sep_argument(rootRegion);
}

} // namespace jlm::hls

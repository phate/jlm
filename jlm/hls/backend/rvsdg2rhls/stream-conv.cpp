//
// Created by david on 7/2/21.
//

#include "jlm/hls/ir/hls.hpp"
#include "rhls-dne.hpp"
#include <algorithm>
#include <jlm/hls/backend/rvsdg2rhls/hls-function-util.hpp>
#include <jlm/hls/backend/rvsdg2rhls/stream-conv.hpp>
#include <jlm/rvsdg/lambda.hpp>

namespace jlm::hls
{

void
stream_conv(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = &graph.GetRootRegion();
  auto lambda = dynamic_cast<rvsdg::LambdaNode *>(root->Nodes().begin().ptr());
  auto stream_enqs = find_function_arguments(lambda, "hls_stream_enq");
  auto stream_deqs = find_function_arguments(lambda, "hls_stream_deq");
  if (stream_enqs.empty())
  {
    JLM_ASSERT(stream_deqs.empty());
    return;
  }
  std::vector<rvsdg::SimpleNode *> enq_calls, deq_calls;
  std::unordered_set<rvsdg::output *> visited;
  for (auto stream_enq : stream_enqs)
  {
    JLM_ASSERT(stream_enq.inner);
    trace_function_calls(stream_enq.inner, enq_calls, visited);
    visited.erase(visited.begin(), visited.end());
  }
  for (auto stream_deq : stream_deqs)
  {
    trace_function_calls(stream_deq.inner, deq_calls, visited);
    visited.erase(visited.begin(), visited.end());
  }
  JLM_ASSERT(!enq_calls.empty());
  JLM_ASSERT(!deq_calls.empty());
  for (auto & enq_call : enq_calls)
  {
    auto enq_constant = trace_constant(enq_call->input(1)->origin());
    for (auto & deq_call : deq_calls)
    {
      auto deq_constant = trace_constant(deq_call->input(1)->origin());
      if (*enq_constant == *deq_constant)
      {
        int buffer_capacity = 10;
        // buffer size as second argument
        if (dynamic_cast<const jlm::rvsdg::bittype *>(&deq_call->input(2)->type()))
        {
          auto constant = trace_constant(deq_call->input(2)->origin());
          buffer_capacity = constant->Representation().to_int();
          JLM_ASSERT(buffer_capacity >= 0);
        }
        auto buf =
            jlm::hls::buffer_op::create(*enq_call->input(2)->origin(), buffer_capacity, false)[0];
        auto routed = route_to_region_rhls(deq_call->region(), buf);
        // remove call nodes
        for (size_t i = 0; i < deq_call->ninputs(); ++i)
        {
          if (dynamic_cast<const rvsdg::StateType *>(&deq_call->input(i)->type()))
          {
            int oi = deq_call->noutputs() - deq_call->ninputs() + i;
            deq_call->output(oi)->divert_users(deq_call->input(i)->origin());
          }
        }
        deq_call->output(0)->divert_users(routed);
        remove(deq_call);
        for (size_t i = 0; i < enq_call->ninputs(); ++i)
        {
          if (dynamic_cast<const rvsdg::StateType *>(&enq_call->input(i)->type()))
          {
            int oi = enq_call->noutputs() - enq_call->ninputs() + i;
            enq_call->output(oi)->divert_users(enq_call->input(i)->origin());
          }
        }
        remove(enq_call);
        // remove deq_call from list
        deq_calls.erase(std::find(deq_calls.begin(), deq_calls.end(), deq_call));
        break;
      }
    }
  }
  // clean up routed function pointers
  dne(lambda->subregion());
  std::vector<rvsdg::LambdaNode::ContextVar> remove_vars(stream_enqs);
  remove_vars.insert(remove_vars.cend(), stream_deqs.begin(), stream_deqs.end());
  // make sure context vars are actually dead
  for (auto cv : remove_vars)
  {
    JLM_ASSERT(cv.inner->nusers() == 0);
  }
  // remove dead cvargs
  lambda->PruneLambdaInputs();
}
}

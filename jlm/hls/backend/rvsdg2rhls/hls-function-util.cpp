//
// Created by david on 7/2/21.
//

#include <jlm/hls/backend/rvsdg2rhls/hls-function-util.hpp>
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

namespace jlm::hls
{

std::vector<rvsdg::LambdaNode::ContextVar>
find_function_arguments(const rvsdg::LambdaNode * lambda, std::string name_contains)
{
  std::vector<rvsdg::LambdaNode::ContextVar> result;
  for (auto cv : lambda->GetContextVars())
  {
    auto ip = cv.input;
    auto traced = trace_call_rhls(ip);
    JLM_ASSERT(traced);
    auto arg = util::AssertedCast<const llvm::GraphImport>(traced);
    if (dynamic_cast<const rvsdg::FunctionType *>(arg->ImportedType().get())
        && arg->Name().find(name_contains) != arg->Name().npos)
    {
      result.push_back(cv);
    }
  }
  return result;
}

void
trace_function_calls(
    rvsdg::Output * output,
    std::vector<rvsdg::SimpleNode *> & calls,
    std::unordered_set<rvsdg::Output *> & visited)
{
  if (visited.count(output))
  {
    // skip already processed outputs
    return;
  }
  visited.insert(output);
  for (auto & user : output->Users())
  {
    if (auto simplenode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(user))
    {
      if (dynamic_cast<const llvm::CallOperation *>(&simplenode->GetOperation()))
      {
        // TODO: verify this is the right type of function call
        calls.push_back(simplenode);
      }
      else
      {
        for (size_t i = 0; i < simplenode->noutputs(); ++i)
        {
          trace_function_calls(simplenode->output(i), calls, visited);
        }
      }
    }
    else if (auto sti = dynamic_cast<rvsdg::StructuralInput *>(&user))
    {
      for (auto & arg : sti->arguments)
      {
        trace_function_calls(&arg, calls, visited);
      }
    }
    else if (auto r = dynamic_cast<rvsdg::RegionResult *>(&user))
    {
      if (auto ber = dynamic_cast<BackEdgeResult *>(r))
      {
        trace_function_calls(ber->argument(), calls, visited);
      }
      else
      {
        trace_function_calls(r->output(), calls, visited);
      }
    }
    else
    {
      JLM_UNREACHABLE("THIS SHOULD BE COVERED");
    }
  }
}

const llvm::IntegerConstantOperation *
trace_constant(const rvsdg::Output * dst)
{
  if (auto arg = dynamic_cast<const rvsdg::RegionArgument *>(dst))
  {
    return trace_constant(arg->input()->origin());
  }

  auto [constantNode, constantOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::IntegerConstantOperation>(*dst);
  if (constantNode)
  {
    if (constantOperation)
      return constantOperation;

    for (size_t i = 0; i < constantNode->ninputs(); ++i)
    {
      // TODO: fix, this is a hack - only works because of distribute constants
      if (*constantNode->input(i)->Type() == *dst->Type())
      {
        return trace_constant(constantNode->input(i)->origin());
      }
    }
  }

  JLM_UNREACHABLE("Constant not found");
}

rvsdg::Output *
route_to_region_rhls(rvsdg::Region * target, rvsdg::Output * out)
{
  // create lists of nested regions
  std::deque<rvsdg::Region *> target_regions = get_parent_regions(target);
  std::deque<rvsdg::Region *> out_regions = get_parent_regions(out->region());
  JLM_ASSERT(target_regions.front() == out_regions.front());
  // remove common ancestor regions
  rvsdg::Region * common_region = nullptr;
  while (!target_regions.empty() && !out_regions.empty()
         && target_regions.front() == out_regions.front())
  {
    common_region = target_regions.front();
    target_regions.pop_front();
    out_regions.pop_front();
  }
  // route out to convergence point from out
  rvsdg::Output * common_out = route_request_rhls(common_region, out);
  auto common_loop = dynamic_cast<LoopNode *>(common_region->node());
  if (common_loop)
  {
    // add a backedge to prevent cycles
    auto arg = common_loop->add_backedge(out->Type());
    arg->result()->divert_to(common_out);
    // route inwards from convergence point to target
    auto result = route_response_rhls(target, arg);
    return result;
  }
  else
  {
    // lambda is common region - might create cycle
    // TODO: how to check that this won't create a cycle
    JLM_ASSERT(
        target_regions.empty() || target_regions.front()->node()->region() == common_out->region());
    return route_response_rhls(target, common_out);
  }
}

rvsdg::Output *
route_response_rhls(rvsdg::Region * target, rvsdg::Output * response)
{
  if (response->region() == target)
  {
    return response;
  }
  else
  {
    auto parent_response = route_response_rhls(target->node()->region(), response);
    auto ln = util::AssertedCast<LoopNode>(target->node());
    return ln->addResponseInput(parent_response);
  }
}

rvsdg::Output *
route_request_rhls(rvsdg::Region * target, rvsdg::Output * request)
{
  if (request->region() == target)
  {
    return request;
  }

  auto ln = util::AssertedCast<LoopNode>(request->region()->node());
  auto output = ln->addRequestOutput(request);

  return route_request_rhls(target, output);
}

std::deque<rvsdg::Region *>
get_parent_regions(rvsdg::Region * region)
{
  std::deque<rvsdg::Region *> regions;
  rvsdg::Region * target_region = region;
  while (!dynamic_cast<const llvm::LlvmLambdaOperation *>(&target_region->node()->GetOperation()))
  {
    regions.push_front(target_region);
    target_region = target_region->node()->region();
  }
  regions.push_front(target_region);
  return regions;
}

const rvsdg::Output *
trace_call_rhls(const rvsdg::Output * output)
{
  // version of trace call for rhls
  if (auto argument = dynamic_cast<const rvsdg::RegionArgument *>(output))
  {
    auto graph = output->region()->graph();
    if (argument->region() == &graph->GetRootRegion())
    {
      return argument;
    }
    else if (dynamic_cast<const BackEdgeArgument *>(argument))
    {
      // don't follow backedges to avoid cycles
      return nullptr;
    }
    return trace_call_rhls(argument->input());
  }
  else if (auto so = dynamic_cast<const rvsdg::StructuralOutput *>(output))
  {
    for (auto & r : so->results)
    {
      if (auto result = trace_call_rhls(&r))
      {
        return result;
      }
    }
  }
  else if (auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*output))
  {
    for (size_t i = 0; i < simpleNode->ninputs(); ++i)
    {
      auto ip = simpleNode->input(i);
      if (*ip->Type() == *output->Type())
      {
        if (auto result = trace_call_rhls(ip))
        {
          return result;
        }
      }
    }
  }
  else
  {
    JLM_UNREACHABLE("");
  }
  return nullptr;
}

const rvsdg::Output *
trace_call_rhls(const rvsdg::Input * input)
{
  // version of trace call for rhls
  return trace_call_rhls(input->origin());
}

bool
is_function_argument(const rvsdg::LambdaNode::ContextVar & cv)
{
  auto ip = cv.input;
  auto traced = trace_call_rhls(ip);
  JLM_ASSERT(traced);
  auto arg = util::AssertedCast<const llvm::GraphImport>(traced);
  return dynamic_cast<const rvsdg::FunctionType *>(arg->ImportedType().get());
}

std::string
get_function_name(jlm::rvsdg::Input * input)
{
  auto traced = jlm::hls::trace_call_rhls(input);
  JLM_ASSERT(traced);
  auto arg = jlm::util::AssertedCast<const jlm::llvm::GraphImport>(traced);
  return arg->Name();
}

bool
is_dec_req(rvsdg::SimpleNode * node)
{
  if (dynamic_cast<const llvm::CallOperation *>(&node->GetOperation()))
  {
    auto name = get_function_name(node->input(0));
    if (name.rfind("decouple_req") != name.npos)
      return true;
  }
  return false;
}

bool
is_dec_res(rvsdg::SimpleNode * node)
{
  if (dynamic_cast<const llvm::CallOperation *>(&node->GetOperation()))
  {
    auto name = get_function_name(node->input(0));
    if (name.rfind("decouple_res") != name.npos)
      return true;
  }
  return false;
}

rvsdg::Input *
get_mem_state_user(rvsdg::Output * state_edge)
{
  JLM_ASSERT(state_edge);
  JLM_ASSERT(state_edge->nusers() == 1);
  JLM_ASSERT(rvsdg::is<llvm::MemoryStateType>(state_edge->Type()));
  return &state_edge->SingleUser();
}

rvsdg::Output *
FindSourceNode(rvsdg::Output * out)
{
  if (auto ba = dynamic_cast<BackEdgeArgument *>(out))
  {
    return FindSourceNode(ba->result()->origin());
  }
  else if (auto ra = dynamic_cast<rvsdg::RegionArgument *>(out))
  {
    if (ra->input() && rvsdg::TryGetOwnerNode<LoopNode>(*ra->input()))
    {
      return FindSourceNode(ra->input()->origin());
    }
    else
    {
      // lambda argument
      return ra;
    }
  }
  else if (auto so = dynamic_cast<rvsdg::StructuralOutput *>(out))
  {
    JLM_ASSERT(rvsdg::TryGetOwnerNode<LoopNode>(*out));
    return FindSourceNode(so->results.begin()->origin());
  }

  JLM_ASSERT(rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*out));
  return out;
}
}

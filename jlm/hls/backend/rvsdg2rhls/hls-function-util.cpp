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
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/MatchVariant.hpp>
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
    auto arg = util::assertedCast<const llvm::LlvmGraphImport>(traced);
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
    rvsdg::MatchVariant(
        user.GetOwner(),
        [&](rvsdg::Node * node)
        {
          rvsdg::MatchTypeOrFail(
              *node,
              [&](rvsdg::SimpleNode & simplenode)
              {
                rvsdg::MatchTypeWithDefault(
                    simplenode.GetOperation(),
                    [&](const llvm::CallOperation &)
                    {
                      // TODO: verify this is the right type of function call
                      calls.push_back(&simplenode);
                    },
                    [&]()
                    {
                      for (size_t i = 0; i < simplenode.noutputs(); ++i)
                      {
                        trace_function_calls(simplenode.output(i), calls, visited);
                      }
                    });
              },
              [&](LoopNode & loop)
              {
                trace_function_calls(loop.mapInput(user).inner, calls, visited);
              },
              [&](rvsdg::ThetaNode & theta)
              {
                trace_function_calls(theta.MapInputLoopVar(user).pre, calls, visited);
              },
              [&](rvsdg::GammaNode & gamma)
              {
                rvsdg::MatchVariant(
                    gamma.MapInput(user),
                    [&](const rvsdg::GammaNode::MatchVar &)
                    {
                    },
                    [&](const rvsdg::GammaNode::EntryVar & evar)
                    {
                      for (auto out : evar.branchArgument)
                      {
                        trace_function_calls(out, calls, visited);
                      }
                    });
              });
        },
        [&](rvsdg::Region * region)
        {
          rvsdg::MatchTypeOrFail(
              *region->node(),
              [&](LoopNode & loop)
              {
                rvsdg::MatchVariant(
                    loop.mapResult(user),
                    [&](const LoopNode::BackEdgeVar & backedge)
                    {
                      trace_function_calls(backedge.pre, calls, visited);
                    },
                    [&](const LoopNode::ExitVar & exit)
                    {
                      trace_function_calls(exit.output, calls, visited);
                    });
              },
              [&](rvsdg::ThetaNode & theta)
              {
                rvsdg::MatchVariant(
                    theta.mapResult(user),
                    [&](const rvsdg::ThetaNode::LoopVar & loopvar)
                    {
                      trace_function_calls(loopvar.output, calls, visited);
                    },
                    [&](const rvsdg::ThetaNode::PredicateVar &)
                    {
                    });
              },
              [&](rvsdg::GammaNode & gamma)
              {
                trace_function_calls(gamma.MapBranchResultExitVar(user).output, calls, visited);
              });
        });
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
    auto ln = util::assertedCast<LoopNode>(target->node());
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

  auto ln = util::assertedCast<LoopNode>(request->region()->node());
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
  return rvsdg::MatchVariant(
      output->GetOwner(),
      [&](rvsdg::Region * region) -> const rvsdg::Output *
      {
        if (region->IsRootRegion())
        {
          return output;
        }
        else if (dynamic_cast<const BackEdgeArgument *>(output))
        {
          // don't follow backedges to avoid cycles
          return nullptr;
        }
        return trace_call_rhls(dynamic_cast<const rvsdg::RegionArgument *>(output)->input());
      },
      [&](rvsdg::Node * node) -> const rvsdg::Output *
      {
        if (auto so = dynamic_cast<const rvsdg::StructuralOutput *>(output))
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
          for (auto & input : simpleNode->Inputs())
          {
            if (*input.Type() == *output->Type())
            {
              if (auto result = trace_call_rhls(&input))
              {
                return result;
              }
            }
          }
        }
        return nullptr;
      });
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
  auto arg = util::assertedCast<const llvm::LlvmGraphImport>(traced);
  return dynamic_cast<const rvsdg::FunctionType *>(arg->ImportedType().get());
}

std::string
get_function_name(jlm::rvsdg::Input * input)
{
  auto traced = jlm::hls::trace_call_rhls(input);
  JLM_ASSERT(traced);
  auto arg = jlm::util::assertedCast<const jlm::llvm::LlvmGraphImport>(traced);
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

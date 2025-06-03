//
// Created by david on 7/2/21.
//

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_HLS_FUNCTION_UTIL_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_HLS_FUNCTION_UTIL_HPP

#include "jlm/llvm/ir/operators/IntegerOperations.hpp"
#include <deque>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/lambda.hpp>

namespace jlm::hls
{
bool
is_function_argument(const rvsdg::LambdaNode::ContextVar & cv);

std::vector<rvsdg::LambdaNode::ContextVar>
find_function_arguments(const rvsdg::LambdaNode * lambda, std::string name_contains);

void
trace_function_calls(
    rvsdg::Output * output,
    std::vector<rvsdg::SimpleNode *> & calls,
    std::unordered_set<rvsdg::Output *> & visited);

const llvm::IntegerConstantOperation *
trace_constant(const rvsdg::Output * dst);

rvsdg::Output *
route_to_region_rhls(rvsdg::Region * target, rvsdg::Output * out);

rvsdg::Output *
route_response_rhls(rvsdg::Region * target, rvsdg::Output * response);

rvsdg::Output *
route_request_rhls(rvsdg::Region * target, rvsdg::Output * request);

std::deque<rvsdg::Region *>
get_parent_regions(rvsdg::Region * region);

const rvsdg::Output *
trace_call_rhls(const rvsdg::Input * input);

const rvsdg::Output *
trace_call_rhls(const rvsdg::Output * output);

std::string
get_function_name(jlm::rvsdg::Input * input);

// this might already exist somewhere
template<typename OpType>
inline const OpType *
TryGetOwnerOp(const rvsdg::Input & input) noexcept
{
  if (const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(input))
  {
    return dynamic_cast<const OpType *>(&node->GetOperation());
  }
  else
  {
    return nullptr;
  }
}

template<typename OpType>
inline const OpType *
TryGetOwnerOp(const rvsdg::Output & output) noexcept
{
  if (const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
  {
    return dynamic_cast<const OpType *>(&node->GetOperation());
  }
  else
  {
    return nullptr;
  }
}

bool
is_dec_req(rvsdg::SimpleNode * node);

bool
is_dec_res(rvsdg::SimpleNode * node);

rvsdg::Input *
get_mem_state_user(rvsdg::Output * state_edge);

/**
 * Traces the origin of the given RVSDG output to find the original source of the value, which is
 * either the output of a SimpleNode, or a function argument.
 *
 * Assumes no gamma or theta nodes are present.
 *
 * @param out The output to be traced to its source
 */
rvsdg::Output *
FindSourceNode(rvsdg::Output * out);
}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_HLS_FUNCTION_UTIL_HPP

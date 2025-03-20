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
find_function_arguments(const rvsdg::LambdaNode * lambda, std::string name_prefix);;

void
trace_function_calls(
    rvsdg::output * output,
    std::vector<rvsdg::SimpleNode *> & calls,
    std::unordered_set<rvsdg::output *> & visited);


const llvm::IntegerConstantOperation *
trace_constant(const rvsdg::output * dst);

rvsdg::output *
route_to_region_rhls(rvsdg::Region * target, rvsdg::output * out);

rvsdg::output *
route_response_rhls(rvsdg::Region * target, rvsdg::output * response);

rvsdg::output *
route_request_rhls(rvsdg::Region * target, rvsdg::output * request);

std::deque<rvsdg::Region *>
get_parent_regions(rvsdg::Region * region);

const rvsdg::output *
trace_call_rhls(const rvsdg::input * input);

const rvsdg::output *
trace_call_rhls(const rvsdg::output * output);
}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_HLS_FUNCTION_UTIL_HPP

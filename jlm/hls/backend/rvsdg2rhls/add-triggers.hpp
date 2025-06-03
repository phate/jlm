/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_ADD_TRIGGERS_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_ADD_TRIGGERS_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jlm::hls
{

rvsdg::Output *
get_trigger(rvsdg::Region * region);

rvsdg::LambdaNode *
add_lambda_argument(rvsdg::LambdaNode * ln, const rvsdg::Type * type);

void
add_triggers(rvsdg::Region * region);

void
add_triggers(llvm::RvsdgModule & rm);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_ADD_TRIGGERS_HPP

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

rvsdg::output *
get_trigger(rvsdg::Region * region);

llvm::lambda::node *
add_lambda_argument(llvm::lambda::node * ln, const rvsdg::type * type);

void
add_triggers(rvsdg::Region * region);

void
add_triggers(llvm::RvsdgModule & rm);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_ADD_TRIGGERS_HPP

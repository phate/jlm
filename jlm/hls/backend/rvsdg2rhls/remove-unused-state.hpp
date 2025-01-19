/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jlm::hls
{

bool
is_passthrough(const rvsdg::output * arg);

bool
is_passthrough(const rvsdg::input * res);

llvm::lambda::node *
remove_lambda_passthrough(llvm::lambda::node * ln);

} // namespace jlm::hls

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP

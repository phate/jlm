/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LIBJLM_SRC_BACKEND_HLS_RVSDG2RHLS_DISTRIBUTE_CONSTANTS_HPP
#define JLM_LIBJLM_SRC_BACKEND_HLS_RVSDG2RHLS_DISTRIBUTE_CONSTANTS_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jlm
{
namespace hls
{
void
distribute_constants(rvsdg::Region * region);

void
distribute_constants(llvm::RvsdgModule & rm);
}
}
#endif // JLM_LIBJLM_SRC_BACKEND_HLS_RVSDG2RHLS_DISTRIBUTE_CONSTANTS_HPP

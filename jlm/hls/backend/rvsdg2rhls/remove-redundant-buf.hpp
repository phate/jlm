/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_REMOVE_REDUNDANT_BUF_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_REMOVE_REDUNDANT_BUF_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>

namespace jlm::hls
{

void
remove_redundant_buf(llvm::RvsdgModule & rm);

void
remove_redundant_buf(rvsdg::Region * region);

}

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_REMOVE_REDUNDANT_BUF_HPP

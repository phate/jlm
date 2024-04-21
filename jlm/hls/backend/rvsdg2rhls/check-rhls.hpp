/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_CHECK_RHLS_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_CHECK_RHLS_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

void
check_rhls(jlm::rvsdg::region * sr);

void
check_rhls(llvm::RvsdgModule & rm);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_CHECK_RHLS_HPP

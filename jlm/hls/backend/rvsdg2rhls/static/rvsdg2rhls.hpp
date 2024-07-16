/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_STATIC_RVSDG2RHLS_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_STATIC_RVSDG2RHLS_HPP

#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::static_hls
{

void
rvsdg2rhls(llvm::RvsdgModule & rvsdgModule);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_STATIC_RVSDG2RHLS_HPP
/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_ADD_PRINTS_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_ADD_PRINTS_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jlm::hls
{

void
add_prints(rvsdg::Region * region);

void
add_prints(llvm::LlvmRvsdgModule & rm);

void
convert_prints(llvm::LlvmRvsdgModule & rm);

void
convert_prints(
    rvsdg::Region * region,
    rvsdg::Output * printf,
    const std::shared_ptr<const rvsdg::FunctionType> & functionType);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_ADD_PRINTS_HPP

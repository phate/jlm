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
add_prints(llvm::RvsdgModule & rm);

void
convert_prints(llvm::RvsdgModule & rm);

void
convert_prints(
    rvsdg::Region * region,
    rvsdg::output * printf,
    const std::shared_ptr<const rvsdg::FunctionType> & functionType);

rvsdg::output *
route_to_region(rvsdg::output * output, rvsdg::Region * region);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_ADD_PRINTS_HPP

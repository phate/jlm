/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_INSTRUMENT_REF_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_INSTRUMENT_REF_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jlm::hls
{

void
instrument_ref(llvm::RvsdgModule & rm);

void
instrument_ref(
    jlm::rvsdg::region * region,
    jlm::rvsdg::output * ioState,
    jlm::rvsdg::output * load_func,
    const llvm::FunctionType & loadFunctionType,
    jlm::rvsdg::output * store_func,
    const llvm::FunctionType & storeFunctionType,
    jlm::rvsdg::output * alloca_func,
    const llvm::FunctionType & allocaFunctionType);

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_INSTRUMENT_REF_HPP

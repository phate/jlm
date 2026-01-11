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
instrument_ref(llvm::LlvmRvsdgModule & rm);

void
instrument_ref(
    rvsdg::Region * region,
    jlm::rvsdg::Output * ioState,
    jlm::rvsdg::Output * load_func,
    const std::shared_ptr<const rvsdg::FunctionType> & loadFunctionType,
    jlm::rvsdg::Output * store_func,
    const std::shared_ptr<const rvsdg::FunctionType> & storeFunctionType,
    jlm::rvsdg::Output * alloca_func,
    const std::shared_ptr<const rvsdg::FunctionType> & allocaFunctionType);

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_INSTRUMENT_REF_HPP

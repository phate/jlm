/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_STATIC_THETACONVERSION_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_STATIC_THETACONVERSION_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/hls/ir/static-hls.hpp>

namespace jlm::static_hls
{

void
ConvertThetaNodes(jlm::llvm::RvsdgModule & rvsdgModule);

} // namespace jlm::static_hls

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_STATIC_THETACONVERSION_HPP
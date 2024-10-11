//
// Created by david on 7/2/21.
//

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_DAE_CONV_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_DAE_CONV_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

void
dae_conv(rvsdg::Region * region);

void
dae_conv(llvm::RvsdgModule & rm);

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_DAE_CONV_HPP

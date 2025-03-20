//
// Created by david on 7/2/21.
//

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_STREAM_CONV_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_STREAM_CONV_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

void
stream_conv(llvm::RvsdgModule & rm);

}
#endif // JLM_HLS_BACKEND_RVSDG2RHLS_STREAM_CONV_HPP

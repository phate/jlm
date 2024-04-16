//
// Created by david on 7/2/21.
//

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_MEM_QUEUE_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_MEM_QUEUE_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

void
mem_queue(jlm::rvsdg::region * region);

void
mem_queue(llvm::RvsdgModule & rm);

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_MEM_QUEUE_HPP

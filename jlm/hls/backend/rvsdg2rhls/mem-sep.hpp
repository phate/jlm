//
// Created by david on 7/2/21.
//

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_MEM_SEP_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_MEM_SEP_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

void
mem_sep_independent(rvsdg::Region * region);

void
mem_sep_independent(llvm::RvsdgModule & rm);

void
mem_sep_argument(rvsdg::Region * region);

void
mem_sep_argument(llvm::RvsdgModule & rm);

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_MEM_SEP_HPP

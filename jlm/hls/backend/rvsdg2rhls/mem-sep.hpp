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
mem_sep_independent(jlm::rvsdg::region * region);

void
mem_sep_independent(llvm::RvsdgModule & rm);

void
mem_sep_argument(jlm::rvsdg::region * region);

void
mem_sep_argument(llvm::RvsdgModule & rm);

rvsdg::RegionArgument *
GetMemoryStateArgument(const llvm::lambda::node & lambda);

jlm::rvsdg::result *
GetMemoryStateResult(const llvm::lambda::node & lambda);

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_MEM_SEP_HPP

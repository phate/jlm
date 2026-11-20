/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_MODREFSETFOLDING_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_MODREFSETFOLDING_HPP

#include <memory>

namespace jlm::rvsdg
{
class Region;
class RvsdgModule;
}

namespace jlm::llvm::aa
{

class MemoryStateGraph;
class ModRefSummary;

std::unique_ptr<MemoryStateGraph>
constructMemoryStateGraph(const rvsdg::Region & region, const ModRefSummary & modRefSummary);

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_MODREFSETFOLDING_HPP

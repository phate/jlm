/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_MODREFSETFOLDING_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_MODREFSETFOLDING_HPP

namespace jlm::rvsdg
{
class RvsdgModule;
}

namespace jlm::llvm::aa
{

class ModRefSummary;

void
foldModRefSets(ModRefSummary & modRefSummary, const rvsdg::RvsdgModule & rvsdgModule);

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_MODREFSETFOLDING_HPP

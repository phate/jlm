/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_UNUSEDSTATEREMOVAL_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_UNUSEDSTATEREMOVAL_HPP

namespace jlm::llvm
{
class RvsdgModule;
}

namespace jlm::hls
{

void
RemoveUnusedStates(llvm::RvsdgModule & rvsdgModule);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_UNUSEDSTATEREMOVAL_HPP

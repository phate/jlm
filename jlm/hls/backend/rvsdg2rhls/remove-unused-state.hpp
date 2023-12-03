/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP

namespace jlm::llvm
{
class RvsdgModule;
}

namespace jlm::hls
{

void
remove_unused_state(llvm::RvsdgModule & rvsdgModule);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP

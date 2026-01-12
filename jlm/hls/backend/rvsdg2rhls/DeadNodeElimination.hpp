/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_DEADNODEELIMINATION_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_DEADNODEELIMINATION_HPP

#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

/**
 * Removes dead loop nodes and their outputs and inputs.
 *
 * @param rvsdgModule The RVSDG module the transformation is performed on.
 *
 * FIXME: This code should be incorporated into llvm::DeadNodeElimination. However, before this
 * can happen, llvm::DeadNodeElimination needs to be moved into the rvsdg namespace and made
 * extensible such that transformation users can register clean up functions for structural nodes.
 *
 * \see hls::loop_node
 */
void
EliminateDeadNodes(llvm::LlvmRvsdgModule & rvsdgModule);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_DEADNODEELIMINATION_HPP

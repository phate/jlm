/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_FRONTEND_CONTROLFLOWRESTRUCTURING_HPP
#define JLM_LLVM_FRONTEND_CONTROLFLOWRESTRUCTURING_HPP

namespace jlm::llvm
{

class ControlFlowGraph;

void
RestructureLoops(ControlFlowGraph * cfg);

void
RestructureBranches(ControlFlowGraph * cfg);

void
RestructureControlFlow(ControlFlowGraph * cfg);

}

#endif

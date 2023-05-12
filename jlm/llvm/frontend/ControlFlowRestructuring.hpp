/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_FRONTEND_CONTROLFLOWRESTRUCTURING_HPP
#define JLM_LLVM_FRONTEND_CONTROLFLOWRESTRUCTURING_HPP

namespace jlm::llvm
{

class cfg;

void
RestructureLoops(llvm::cfg * cfg);

void
RestructureBranches(llvm::cfg * cfg);

void
RestructureControlFlow(llvm::cfg * cfg);

}

#endif

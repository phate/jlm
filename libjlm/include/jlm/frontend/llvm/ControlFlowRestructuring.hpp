/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_FRONTEND_LLVM_JLM2RVSDG_CONTROLFLOWRESTRUCTURING_HPP
#define JLM_FRONTEND_LLVM_JLM2RVSDG_CONTROLFLOWRESTRUCTURING_HPP

namespace jlm {

class cfg;

void
restructure_loops(jlm::cfg * cfg);

void
RestructureBranches(jlm::cfg * cfg);

void
RestructureControlFlow(jlm::cfg * cfg);

}

#endif

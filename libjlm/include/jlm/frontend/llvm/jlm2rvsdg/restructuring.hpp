/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_FRONTEND_LLVM_JLM2RVSDG_RESTRUCTURING_HPP
#define JLM_FRONTEND_LLVM_JLM2RVSDG_RESTRUCTURING_HPP

#include <unordered_set>

namespace jlm {

class cfg;
class cfg_edge;

void
restructure_loops(jlm::cfg * cfg);

void
restructure_branches(jlm::cfg * cfg);

void
restructure(jlm::cfg * cfg);

}

#endif

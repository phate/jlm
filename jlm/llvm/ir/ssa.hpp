/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_SSA_HPP
#define JLM_LLVM_IR_SSA_HPP

namespace jlm {

class cfg;

void
destruct_ssa(jlm::cfg & cfg);

}

#endif

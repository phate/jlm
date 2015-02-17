/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_CONSTRUCTION_JLM_HPP
#define JLM_CONSTRUCTION_JLM_HPP

#include <jlm/IR/clg.hpp>

#include <unordered_map>

namespace llvm {
	class Module;
}

namespace jlm {

void
convert_module(const llvm::Module & module, jlm::frontend::clg & clg);

}

#endif

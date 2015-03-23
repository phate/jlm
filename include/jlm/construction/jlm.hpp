/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_CONSTRUCTION_JLM_HPP
#define JLM_CONSTRUCTION_JLM_HPP

#include <unordered_map>

namespace llvm {
	class Module;
}

namespace jlm {

class clg;

void
convert_module(const llvm::Module & module, jlm::clg & clg);

}

#endif

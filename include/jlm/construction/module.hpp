/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_CONSTRUCTION_MODULE_HPP
#define JLM_CONSTRUCTION_MODULE_HPP

namespace llvm {
	class Module;
}

namespace jlm {

class module;

void
convert_module(const llvm::Module & module, jlm::module & clg);

}

#endif

/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_FRONTEND_LLVM_LLVM2JLM_MODULE_HPP
#define JLM_FRONTEND_LLVM_LLVM2JLM_MODULE_HPP

#include <memory>

namespace llvm {
	class Module;
}

namespace jlm {

class ipgraph_module;

std::unique_ptr<ipgraph_module>
convert_module(llvm::Module & module);

}

#endif

/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM2JLM_MODULE_HPP
#define JLM_LLVM2JLM_MODULE_HPP

#include <memory>

namespace llvm {
	class Module;
}

namespace jlm {

class module;

std::unique_ptr<module>
convert_module(const llvm::Module & module);

}

#endif

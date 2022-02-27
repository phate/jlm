/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_FRONTEND_LLVM_LLVM2JLM_LLVMMODULECONVERSION_HPP
#define JLM_FRONTEND_LLVM_LLVM2JLM_LLVMMODULECONVERSION_HPP

#include <jlm/ir/attribute.hpp>

#include <llvm/IR/Attributes.h>

#include <memory>

namespace llvm {
	class Module;
}

namespace jlm {

class ipgraph_module;

attribute::kind
convert_attribute_kind(const llvm::Attribute::AttrKind & kind);

std::unique_ptr<ipgraph_module>
ConvertLlvmModule(llvm::Module & module);

}

#endif

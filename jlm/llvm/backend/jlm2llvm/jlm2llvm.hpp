/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_BACKEND_JLM2LLVM_JLM2LLVM_HPP
#define JLM_LLVM_BACKEND_JLM2LLVM_JLM2LLVM_HPP

#include <jlm/llvm/ir/attribute.hpp>

#include <llvm/IR/Attributes.h>

#include <memory>

namespace llvm {

class LLVMContext;
class Module;

}

namespace jlm {

class ipgraph_module;

namespace jlm2llvm {

llvm::Attribute::AttrKind
convert_attribute_kind(const jlm::attribute::kind & kind);

/*
	FIXME: ipgraph_module should be const, but we still need to create variables to translate
	       expressions.
*/
std::unique_ptr<llvm::Module>
convert(ipgraph_module & im, llvm::LLVMContext & ctx);

}}

#endif

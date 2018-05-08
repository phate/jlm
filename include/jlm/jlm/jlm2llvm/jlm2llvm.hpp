/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLM2LLVM_JLM2LLVM_HPP
#define JLM_JLM2LLVM_JLM2LLVM_HPP

#include <memory>

namespace llvm {

class LLVMContext;
class Module;

}

namespace jlm {
namespace jlm2llvm {

class module;
/*
	FIXME: jlm::module should be const, but we still need to create variables to translate
	       expressions.
*/
std::unique_ptr<llvm::Module>
convert(jlm::module & module, llvm::LLVMContext & ctx);

}}

#endif

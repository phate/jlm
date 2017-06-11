/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLM2LLVM_TYPE_HPP
#define JLM_JLM2LLVM_TYPE_HPP

#include <jive/types/bitstring/type.h>
#include <jive/types/function/fcttype.h>

#include <jlm/common.hpp>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

namespace llvm {

class FunctionType;

}

namespace jlm {
namespace jlm2llvm {

llvm::Type *
convert_type(const jive::base::type & type, llvm::LLVMContext & ctx);

static inline llvm::IntegerType *
convert_type(const jive::bits::type & type, llvm::LLVMContext & ctx)
{
	auto t = convert_type(*static_cast<const jive::base::type*>(&type), ctx);
	JLM_DEBUG_ASSERT(t->getTypeID() == llvm::Type::IntegerTyID);
	return llvm::cast<llvm::IntegerType>(t);
}

static inline llvm::FunctionType *
convert_type(const jive::fct::type & type, llvm::LLVMContext & ctx)
{
	auto t = convert_type(*static_cast<const jive::base::type*>(&type), ctx);
	JLM_DEBUG_ASSERT(t->getTypeID() == llvm::Type::FunctionTyID);
	return llvm::cast<llvm::FunctionType>(t);
}

}}

#endif

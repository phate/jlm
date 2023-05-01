/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_BACKEND_JLM2LLVM_TYPE_HPP
#define JLM_LLVM_BACKEND_JLM2LLVM_TYPE_HPP

#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/record.hpp>
#include <jlm/util/common.hpp>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

namespace llvm {

class FunctionType;

}

namespace jlm {
namespace jlm2llvm {

class context;

llvm::Type *
convert_type(const jive::type & type, context & ctx);

static inline llvm::IntegerType *
convert_type(const jive::bittype & type, context & ctx)
{
	auto t = convert_type(*static_cast<const jive::type*>(&type), ctx);
	JLM_ASSERT(t->getTypeID() == llvm::Type::IntegerTyID);
	return llvm::cast<llvm::IntegerType>(t);
}

static inline llvm::FunctionType *
convert_type(const jlm::FunctionType & type, context & ctx)
{
	auto t = convert_type(*static_cast<const jive::type*>(&type), ctx);
	JLM_ASSERT(t->getTypeID() == llvm::Type::FunctionTyID);
	return llvm::cast<llvm::FunctionType>(t);
}

static inline llvm::PointerType *
convert_type(const jlm::PointerType & type, context & ctx)
{
	auto t = convert_type(*static_cast<const jive::type*>(&type), ctx);
	JLM_ASSERT(t->getTypeID() == llvm::Type::PointerTyID);
	return llvm::cast<llvm::PointerType>(t);
}

static inline llvm::ArrayType *
convert_type(const jlm::arraytype & type, context & ctx)
{
	auto t = convert_type(*static_cast<const jive::type*>(&type), ctx);
	JLM_ASSERT(t->getTypeID() == llvm::Type::ArrayTyID);
	return llvm::cast<llvm::ArrayType>(t);
}

static inline llvm::IntegerType *
convert_type(const jive::ctltype & type, context & ctx)
{
	auto t = convert_type(*static_cast<const jive::type*>(&type), ctx);
	JLM_ASSERT(t->getTypeID() == llvm::Type::IntegerTyID);
	return llvm::cast<llvm::IntegerType>(t);
}

static inline llvm::Type *
convert_type(const jlm::fptype & type, context & ctx)
{
	auto t = convert_type(*static_cast<const jive::type*>(&type), ctx);
	JLM_ASSERT(t->isHalfTy() || t->isFloatTy() || t->isDoubleTy());
	return t;
}

static inline llvm::StructType *
convert_type(const StructType & type, context & ctx)
{
	auto t = convert_type(*static_cast<const jive::type*>(&type), ctx);
	JLM_ASSERT(t->isStructTy());
	return llvm::cast<llvm::StructType>(t);
}

static inline llvm::VectorType *
convert_type(const vectortype & type, context & ctx)
{
	auto t = convert_type(*static_cast<const jive::type*>(&type), ctx);
	JLM_ASSERT(t->isVectorTy());
	return llvm::cast<llvm::VectorType>(t);
}

}}

#endif

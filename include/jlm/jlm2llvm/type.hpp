/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLM2LLVM_TYPE_HPP
#define JLM_JLM2LLVM_TYPE_HPP

#include <jive/types/bitstring/type.h>
#include <jive/types/function/fcttype.h>
#include <jive/types/record/rcdtype.h>
#include <jive/vsdg/controltype.h>

#include <jlm/common.hpp>
#include <jlm/ir/types.hpp>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

namespace llvm {

class FunctionType;

}

namespace jlm {
namespace jlm2llvm {

class context;

llvm::Type *
convert_type(const jive::base::type & type, context & ctx);

static inline llvm::IntegerType *
convert_type(const jive::bits::type & type, context & ctx)
{
	auto t = convert_type(*static_cast<const jive::base::type*>(&type), ctx);
	JLM_DEBUG_ASSERT(t->getTypeID() == llvm::Type::IntegerTyID);
	return llvm::cast<llvm::IntegerType>(t);
}

static inline llvm::FunctionType *
convert_type(const jive::fct::type & type, context & ctx)
{
	auto t = convert_type(*static_cast<const jive::base::type*>(&type), ctx);
	JLM_DEBUG_ASSERT(t->getTypeID() == llvm::Type::FunctionTyID);
	return llvm::cast<llvm::FunctionType>(t);
}

static inline llvm::PointerType *
convert_type(const jlm::ptrtype & type, context & ctx)
{
	auto t = convert_type(*static_cast<const jive::base::type*>(&type), ctx);
	JLM_DEBUG_ASSERT(t->getTypeID() == llvm::Type::PointerTyID);
	return llvm::cast<llvm::PointerType>(t);
}

static inline llvm::ArrayType *
convert_type(const jlm::arraytype & type, context & ctx)
{
	auto t = convert_type(*static_cast<const jive::base::type*>(&type), ctx);
	JLM_DEBUG_ASSERT(t->getTypeID() == llvm::Type::ArrayTyID);
	return llvm::cast<llvm::ArrayType>(t);
}

static inline llvm::IntegerType *
convert_type(const jive::ctl::type & type, context & ctx)
{
	auto t = convert_type(*static_cast<const jive::base::type*>(&type), ctx);
	JLM_DEBUG_ASSERT(t->getTypeID() == llvm::Type::IntegerTyID);
	return llvm::cast<llvm::IntegerType>(t);
}

static inline llvm::Type *
convert_type(const jlm::fptype & type, context & ctx)
{
	auto t = convert_type(*static_cast<const jive::base::type*>(&type), ctx);
	JLM_DEBUG_ASSERT(t->isHalfTy() || t->isFloatTy() || t->isDoubleTy());
	return t;
}

static inline llvm::StructType *
convert_type(const jive::rcd::type & type, context & ctx)
{
	auto t = convert_type(*static_cast<const jive::base::type*>(&type), ctx);
	JLM_DEBUG_ASSERT(t->isStructTy());
	return llvm::cast<llvm::StructType>(t);
}

}}

#endif

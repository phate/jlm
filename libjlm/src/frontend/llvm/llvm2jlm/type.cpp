/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/frontend/llvm/llvm2jlm/context.hpp>
#include <jlm/frontend/llvm/llvm2jlm/type.hpp>
#include <jlm/ir/types.hpp>

#include <jive/arch/addresstype.hpp>
#include <jive/types/bitstring/type.hpp>
#include <jive/types/float/flttype.hpp>
#include <jive/types/function.hpp>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

#include <map>

namespace jlm {

jlm::fpsize
convert_fpsize(const llvm::Type * type)
{
	JLM_ASSERT(type->isFloatingPointTy());

	static std::unordered_map<const llvm::Type::TypeID, jlm::fpsize> map({
	  {llvm::Type::HalfTyID, fpsize::half}
	, {llvm::Type::FloatTyID, fpsize::flt}
	, {llvm::Type::DoubleTyID, fpsize::dbl}
	, {llvm::Type::X86_FP80TyID, fpsize::x86fp80}
	});

	JLM_ASSERT(map.find(type->getTypeID()) != map.end());
	return map[type->getTypeID()];
}

static std::unique_ptr<jive::valuetype>
convert_integer_type(const llvm::Type * t, context & ctx)
{
	JLM_ASSERT(t->getTypeID() == llvm::Type::IntegerTyID);
	const llvm::IntegerType * type = static_cast<const llvm::IntegerType*>(t);

	return std::unique_ptr<jive::valuetype>(new jive::bittype(type->getBitWidth()));
}

static std::unique_ptr<jive::valuetype>
convert_pointer_type(const llvm::Type * t, context & ctx)
{
	JLM_ASSERT(t->getTypeID() == llvm::Type::PointerTyID);
	const auto & type = llvm::cast<llvm::PointerType>(t);

	auto et = convert_type(type->getElementType(), ctx);
	return std::unique_ptr<jive::valuetype>(new jlm::ptrtype(*et));
}

static std::unique_ptr<jive::valuetype>
convert_function_type(const llvm::Type * t, context & ctx)
{
	JLM_ASSERT(t->getTypeID() == llvm::Type::FunctionTyID);
	auto type = llvm::cast<const llvm::FunctionType>(t);

	/* arguments */
	std::vector<std::unique_ptr<jive::type>> argument_types;
	for (size_t n = 0; n < type->getNumParams(); n++)
		argument_types.push_back(convert_type(type->getParamType(n), ctx));
	if (type->isVarArg()) argument_types.push_back(create_varargtype());
	argument_types.push_back(iostatetype::create());
	argument_types.push_back(std::unique_ptr<jive::type>(new jive::memtype()));
	argument_types.push_back(loopstatetype::create());

	/* results */
	std::vector<std::unique_ptr<jive::type>> result_types;
	if (type->getReturnType()->getTypeID() != llvm::Type::VoidTyID)
		result_types.push_back(convert_type(type->getReturnType(), ctx));
	result_types.push_back(iostatetype::create());
	result_types.push_back(std::unique_ptr<jive::type>(new jive::memtype()));
	result_types.push_back(loopstatetype::create());

	return std::unique_ptr<jive::valuetype>(new jive::fcttype(argument_types, result_types));
}

static inline std::unique_ptr<jive::valuetype>
convert_fp_type(const llvm::Type * t, context & ctx)
{
	static std::unordered_map<llvm::Type::TypeID, jlm::fpsize> map({
	  {llvm::Type::HalfTyID, fpsize::half}
	, {llvm::Type::FloatTyID, fpsize::flt}
	, {llvm::Type::DoubleTyID, fpsize::dbl}
	, {llvm::Type::X86_FP80TyID, fpsize::x86fp80}
	});

	JLM_ASSERT(map.find(t->getTypeID()) != map.end());
	return std::unique_ptr<jive::valuetype>(new jlm::fptype(map[t->getTypeID()]));
}

static inline std::unique_ptr<jive::valuetype>
convert_struct_type(const llvm::Type * t, context & ctx)
{
	JLM_ASSERT(t->isStructTy());
	auto type = static_cast<const llvm::StructType*>(t);

	auto packed = type->isPacked();
	auto dcl = ctx.lookup_declaration(type);
	if (type->hasName())
		return std::unique_ptr<jive::valuetype>(new structtype(type->getName().str(), packed, dcl));

	return std::unique_ptr<jive::valuetype>(new structtype(packed, dcl));
}

static std::unique_ptr<jive::valuetype>
convert_array_type(const llvm::Type * t, context & ctx)
{
	JLM_ASSERT(t->isArrayTy());
	auto etype = convert_type(t->getArrayElementType(), ctx);
	return std::unique_ptr<jive::valuetype>(new jlm::arraytype(*etype, t->getArrayNumElements()));
}

static std::unique_ptr<jive::valuetype>
convert_fixed_vector_type(const llvm::Type * t, context & ctx)
{
	JLM_ASSERT(t->getTypeID() == llvm::Type::FixedVectorTyID);
	auto type = convert_type(t->getScalarType(), ctx);
	return std::unique_ptr<jive::valuetype>(new jlm::fixedvectortype(*type, llvm::cast<llvm::FixedVectorType>(t)->getNumElements()));
}

static std::unique_ptr<jive::valuetype>
convert_scalable_vector_type(const llvm::Type * t, context & ctx)
{
	JLM_ASSERT(t->getTypeID() == llvm::Type::ScalableVectorTyID);
	auto type = convert_type(t->getScalarType(), ctx);
	return std::unique_ptr<jive::valuetype>(new jlm::scalablevectortype(*type, llvm::cast<llvm::ScalableVectorType>(t)->getMinNumElements()));
}

std::unique_ptr<jive::valuetype>
convert_type(const llvm::Type * t, context & ctx)
{
	static std::unordered_map<
	  llvm::Type::TypeID
	, std::function<std::unique_ptr<jive::valuetype>(const llvm::Type *, context &)>
	> map({
	  {llvm::Type::IntegerTyID, convert_integer_type}
	, {llvm::Type::PointerTyID, convert_pointer_type}
	, {llvm::Type::FunctionTyID, convert_function_type}
	, {llvm::Type::HalfTyID, convert_fp_type}
	, {llvm::Type::FloatTyID, convert_fp_type}
	, {llvm::Type::DoubleTyID, convert_fp_type}
	, {llvm::Type::X86_FP80TyID, convert_fp_type}
	, {llvm::Type::StructTyID, convert_struct_type}
	, {llvm::Type::ArrayTyID, convert_array_type}
	, {llvm::Type::FixedVectorTyID, convert_fixed_vector_type}
	, {llvm::Type::ScalableVectorTyID, convert_scalable_vector_type}
	});

	JLM_ASSERT(map.find(t->getTypeID()) != map.end());
	return map[t->getTypeID()](t, ctx);
}

}

/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/types.hpp>
#include <jlm/llvm2jlm/context.hpp>
#include <jlm/llvm2jlm/type.hpp>

#include <jive/arch/addresstype.h>
#include <jive/types/bitstring/type.h>
#include <jive/types/float/flttype.h>
#include <jive/types/function/fcttype.h>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

#include <map>

namespace jlm {

static std::unique_ptr<jive::valuetype>
convert_integer_type(const llvm::Type * t, context & ctx)
{
	JLM_DEBUG_ASSERT(t->getTypeID() == llvm::Type::IntegerTyID);
	const llvm::IntegerType * type = static_cast<const llvm::IntegerType*>(t);

	return std::unique_ptr<jive::valuetype>(new jive::bits::type(type->getBitWidth()));
}

static std::unique_ptr<jive::valuetype>
convert_pointer_type(const llvm::Type * t, context & ctx)
{
	JLM_DEBUG_ASSERT(t->getTypeID() == llvm::Type::PointerTyID);
	const auto & type = llvm::cast<llvm::PointerType>(t);

	auto et = convert_type(type->getElementType(), ctx);
	return std::unique_ptr<jive::valuetype>(new jlm::ptrtype(*et));
}

static std::unique_ptr<jive::valuetype>
convert_function_type(const llvm::Type * t, context & ctx)
{
	JLM_DEBUG_ASSERT(t->getTypeID() == llvm::Type::FunctionTyID);
	auto type = llvm::cast<const llvm::FunctionType>(t);

	/* arguments */
	std::vector<std::unique_ptr<jive::type>> argument_types;
	for (size_t n = 0; n < type->getNumParams(); n++)
		argument_types.push_back(convert_type(type->getParamType(n), ctx));
	if (type->isVarArg()) argument_types.push_back(create_varargtype());
	argument_types.push_back(std::unique_ptr<jive::type>(new jive::memtype()));

	/* results */
	std::vector<std::unique_ptr<jive::type>> result_types;
	if (type->getReturnType()->getTypeID() != llvm::Type::VoidTyID)
		result_types.push_back(convert_type(type->getReturnType(), ctx));
	result_types.push_back(std::unique_ptr<jive::type>(new jive::memtype()));

	return std::unique_ptr<jive::valuetype>(new jive::fct::type(argument_types, result_types));
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

	JLM_DEBUG_ASSERT(map.find(t->getTypeID()) != map.end());
	return std::unique_ptr<jive::valuetype>(new jlm::fptype(map[t->getTypeID()]));
}

static inline std::unique_ptr<jive::valuetype>
convert_struct_type(const llvm::Type * t, context & ctx)
{
	JLM_DEBUG_ASSERT(t->isStructTy());
	auto type = static_cast<const llvm::StructType*>(t);

	auto packed = type->isPacked();
	auto decl = ctx.lookup_declaration(type);
	if (type->hasName())
		return std::unique_ptr<jive::valuetype>(new structtype(type->getName(), packed, decl));

	return std::unique_ptr<jive::valuetype>(new structtype(packed, decl));
}

static std::unique_ptr<jive::valuetype>
convert_array_type(const llvm::Type * t, context & ctx)
{
	JLM_DEBUG_ASSERT(t->isArrayTy());
	auto etype = convert_type(t->getArrayElementType(), ctx);
	return std::unique_ptr<jive::valuetype>(new jlm::arraytype(*etype, t->getArrayNumElements()));
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
	});

	JLM_DEBUG_ASSERT(map.find(t->getTypeID()) != map.end());
	return map[t->getTypeID()](t, ctx);
}

}

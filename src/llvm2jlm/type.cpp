/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/types.hpp>
#include <jlm/llvm2jlm/context.hpp>
#include <jlm/llvm2jlm/type.hpp>

#include <jive/arch/addresstype.h>
#include <jive/arch/memorytype.h>
#include <jive/types/bitstring/type.h>
#include <jive/types/float/flttype.h>
#include <jive/types/function/fcttype.h>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

#include <map>

namespace jlm {

static std::unique_ptr<jive::value::type>
convert_integer_type(const llvm::Type * t, context & ctx)
{
	JLM_DEBUG_ASSERT(t->getTypeID() == llvm::Type::IntegerTyID);
	const llvm::IntegerType * type = static_cast<const llvm::IntegerType*>(t);

	return std::unique_ptr<jive::value::type>(new jive::bits::type(type->getBitWidth()));
}

static std::unique_ptr<jive::value::type>
convert_pointer_type(const llvm::Type * t, context & ctx)
{
	JLM_DEBUG_ASSERT(t->getTypeID() == llvm::Type::PointerTyID);
	const auto & type = llvm::cast<llvm::PointerType>(t);

	auto et = convert_type(type->getElementType(), ctx);
	return std::unique_ptr<jive::value::type>(new jlm::ptrtype(*et));
}

static std::unique_ptr<jive::value::type>
convert_function_type(const llvm::Type * t, context & ctx)
{
	JLM_DEBUG_ASSERT(t->getTypeID() == llvm::Type::FunctionTyID);
	const llvm::FunctionType * type = static_cast<const llvm::FunctionType*>(t);

	/* arguments */

	std::vector<std::unique_ptr<jive::base::type>> argument_types;
	for (size_t n = 0; n < type->getNumParams(); n++)
		argument_types.push_back(convert_type(type->getParamType(n), ctx));

	if (type->isVarArg())
		argument_types.push_back(std::unique_ptr<jive::base::type>(new jive::addr::type()));

	argument_types.push_back(std::unique_ptr<jive::base::type>(new jive::mem::type()));

	/* results */

	std::vector<std::unique_ptr<jive::base::type>> result_types;
	if (type->getReturnType()->getTypeID() != llvm::Type::VoidTyID)
		result_types.push_back(convert_type(type->getReturnType(), ctx));
	result_types.push_back(std::unique_ptr<jive::base::type>(new jive::mem::type()));

	return std::unique_ptr<jive::value::type>(new jive::fct::type(argument_types, result_types));
}

static std::unique_ptr<jive::value::type>
convert_float_type(const llvm::Type * t, context & ctx)
{
	JLM_DEBUG_ASSERT(t->isFloatingPointTy());

	/* FIXME: we map all floating point types to float */
	return std::unique_ptr<jive::value::type>(new jive::flt::type());
}

static std::unique_ptr<jive::value::type>
convert_struct_type(const llvm::Type * t, context & ctx)
{
	JLM_DEBUG_ASSERT(t->isStructTy());
	const llvm::StructType * type = static_cast<const llvm::StructType*>(t);

	/* FIXME: handle packed structures */

	return std::unique_ptr<jive::value::type>(new jive::rcd::type(ctx.lookup_declaration(type)));
}

typedef std::map<llvm::Type::TypeID,
	std::unique_ptr<jive::value::type>(*)(const llvm::Type *, context & ctx)> type_map;

static type_map tmap({
		{llvm::Type::IntegerTyID, convert_integer_type}
	, {llvm::Type::PointerTyID, convert_pointer_type}
	, {llvm::Type::FunctionTyID, convert_function_type}
	, {llvm::Type::HalfTyID, convert_float_type}
	, {llvm::Type::FloatTyID, convert_float_type}
	, {llvm::Type::DoubleTyID, convert_float_type}
	, {llvm::Type::X86_FP80TyID, convert_float_type}
	, {llvm::Type::FP128TyID, convert_float_type}
	, {llvm::Type::PPC_FP128TyID, convert_float_type}
	, {llvm::Type::StructTyID, convert_struct_type}
});

std::unique_ptr<jive::value::type>
convert_type(const llvm::Type * t, context & ctx)
{
	JLM_DEBUG_ASSERT(tmap.find(t->getTypeID()) != tmap.end());
	return tmap[t->getTypeID()](t, ctx);
}

}

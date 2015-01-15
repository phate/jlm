/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/type.hpp>

#include <jive/arch/addresstype.h>
#include <jive/arch/memorytype.h>
#include <jive/types/bitstring/type.h>
#include <jive/types/function/fcttype.h>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

#include <map>

namespace jlm {

static std::unique_ptr<jive::base::type>
convert_integer_type(const llvm::Type & t)
{
	const llvm::IntegerType * type = static_cast<const llvm::IntegerType*>(&t);
	JLM_DEBUG_ASSERT(type != nullptr);

	return std::unique_ptr<jive::base::type>(new jive::bits::type(type->getBitWidth()));
}

static std::unique_ptr<jive::base::type>
convert_pointer_type(const llvm::Type & t)
{
	const llvm::PointerType * type = static_cast<const llvm::PointerType*>(&t);
	JLM_DEBUG_ASSERT(type != nullptr);

	return std::unique_ptr<jive::base::type>(new jive::addr::type());
}

static std::unique_ptr<jive::base::type>
convert_function_type(const llvm::Type & t)
{
	const llvm::FunctionType * type = static_cast<const llvm::FunctionType*>(&t);
	JLM_DEBUG_ASSERT(type != nullptr);

	std::vector<std::unique_ptr<jive::base::type>> argument_types;
	for (size_t n = 0; n < type->getNumParams(); n++)
		argument_types.push_back(convert_type(*type->getParamType(n)));
	argument_types.push_back(std::unique_ptr<jive::base::type>(new jive::mem::type()));

	std::vector<std::unique_ptr<jive::base::type>> result_types;
	if (type->getReturnType()->getTypeID() != llvm::Type::VoidTyID)
		result_types.push_back(convert_type(*type->getReturnType()));
	result_types.push_back(std::unique_ptr<jive::base::type>(new jive::mem::type()));

	return std::unique_ptr<jive::base::type>(new jive::fct::type(argument_types, result_types));
}

typedef std::map<llvm::Type::TypeID,
	std::unique_ptr<jive::base::type>(*)(const llvm::Type &)> type_map;

static type_map tmap({
		{llvm::Type::IntegerTyID, convert_integer_type}
	, {llvm::Type::PointerTyID, convert_pointer_type}
	, {llvm::Type::FunctionTyID, convert_function_type}
});

std::unique_ptr<jive::base::type>
convert_type(const llvm::Type & t)
{
	JLM_DEBUG_ASSERT(tmap.find(t.getTypeID()) != tmap.end());
	return tmap[t.getTypeID()](t);
}

}

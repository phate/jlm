/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/type.hpp>

#include <jive/arch/addresstype.h>
#include <jive/types/bitstring/type.h>

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

typedef std::map<llvm::Type::TypeID,
	std::unique_ptr<jive::base::type>(*)(const llvm::Type &)> type_map;

static type_map tmap({
		{llvm::Type::IntegerTyID, convert_integer_type}
	, {llvm::Type::PointerTyID, convert_pointer_type}
});

std::unique_ptr<jive::base::type>
convert_type(const llvm::Type & t)
{
	JLM_DEBUG_ASSERT(tmap.find(t.getTypeID()) != tmap.end());
	return tmap[t.getTypeID()](t);
}

}

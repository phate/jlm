/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/construction/constant.hpp>

#include <jlm/IR/tac/bitstring.hpp>

//FIXME: to be removed, once we have a proper value representation
#include <jive/types/bitstring/value-representation.h>

#include <llvm/IR/Constants.h>

#include <unordered_map>

namespace jlm {

jive::bits::value_repr
convert_apint(const llvm::APInt & value)
{
	std::string str = value.toString(2, value.isNegative());
	std::reverse(str.begin(), str.end());

	jive::bits::value_repr vr(str);
	if (value.isNegative())
		vr.sext(value.getBitWidth() - str.size());
	else
		vr.zext(value.getBitWidth() - str.size());

	return vr;
}

const jlm::frontend::variable *
create_undef_value(const llvm::Type & type, jlm::frontend::basic_block * bb)
{
	if (type.getTypeID() == llvm::Type::IntegerTyID) {
		const llvm::IntegerType * t = static_cast<const llvm::IntegerType*>(&type);
		jive::bits::value_repr v(t->getBitWidth(), 'X');
		return bitconstant_tac(bb, v);
	}

	JLM_DEBUG_ASSERT(0);
	return nullptr;
}

static const jlm::frontend::variable *
convert_int_constant(const llvm::Constant & c, jlm::frontend::basic_block * bb)
{
	const llvm::ConstantInt * constant = static_cast<const llvm::ConstantInt*>(&c);
	JLM_DEBUG_ASSERT(constant != nullptr);

	jive::bits::value_repr v = convert_apint(constant->getValue());
	return bitconstant_tac(bb, v);
}

static const jlm::frontend::variable *
convert_undefvalue_instruction(const llvm::Constant & c, jlm::frontend::basic_block * bb)
{
	const llvm::UndefValue * constant = static_cast<const llvm::UndefValue*>(&c);
	JLM_DEBUG_ASSERT(constant != nullptr);

	return create_undef_value(*constant->getType(), bb);
}

typedef std::unordered_map<
	std::type_index,
	const jlm::frontend::variable*(*)(const llvm::Constant &, jlm::frontend::basic_block*)
	> constant_map;

static constant_map cmap({
		{std::type_index(typeid(llvm::ConstantInt)), convert_int_constant}
	, {std::type_index(typeid(llvm::UndefValue)), convert_undefvalue_instruction}
});

const jlm::frontend::variable *
convert_constant(const llvm::Constant & c, jlm::frontend::basic_block * bb)
{
	JLM_DEBUG_ASSERT(cmap.find(std::type_index(typeid(c))) != cmap.end());
	return cmap[std::type_index(typeid(c))](c, bb);
}

}

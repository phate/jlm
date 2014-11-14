/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/constant.hpp>
#include <jlm/jlm.hpp>

#include <jive/frontend/basic_block.h>
#include <jive/frontend/tac/bitstring.h>

#include <llvm/IR/Instructions.h>

#include <typeindex>

namespace jlm {

const jive::frontend::output *
convert_value(const llvm::Value * v, jive::frontend::basic_block * bb, value_map & vmap)
{
	if (auto c = dynamic_cast<const llvm::Constant*>(v)) {
		const jive::frontend::output * result = convert_constant(*c, bb);
		JLM_DEBUG_ASSERT(vmap.find(v) == vmap.end());
		vmap[v] = result;
	}

	JLM_DEBUG_ASSERT(vmap.find(v) != vmap.end());
	return vmap[v];
}

/* integer binary operators */

typedef std::map<llvm::Instruction::BinaryOps,
	jive::frontend::output *(*)(jive::frontend::basic_block *, size_t, const jive::frontend::output*,
		const jive::frontend::output*)> int_binary_operators_map;

static int_binary_operators_map ibomap({
		{llvm::Instruction::Add, jive::frontend::bitsum_tac}
	,	{llvm::Instruction::And, jive::frontend::bitand_tac}
});

static inline const jive::frontend::output *
convert_int_binary_operator(const llvm::BinaryOperator & i, jive::frontend::basic_block * bb,
	value_map & vmap)
{
	JLM_DEBUG_ASSERT(i.getType()->getTypeID() == llvm::Type::IntegerTyID);

	const llvm::IntegerType * type = static_cast<const llvm::IntegerType*>(i.getType());

	const jive::frontend::output * op1 = convert_value(i.getOperand(0), bb, vmap);
	const jive::frontend::output * op2 = convert_value(i.getOperand(1), bb, vmap);
	return ibomap[i.getOpcode()](bb, type->getBitWidth(), op1, op2);
}

/* instructions */

static void
convert_return_instruction(const llvm::Instruction & i, jive::frontend::basic_block * bb,
	const basic_block_map & bbmap, value_map & vmap)
{
	const llvm::ReturnInst * instruction = static_cast<const llvm::ReturnInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	JLM_DEBUG_ASSERT(instruction->getNumSuccessors() == 0);
	JLM_DEBUG_ASSERT(bb->noutedges() == 0);
	bb->add_outedge(bb->cfg()->exit(), 0);

	convert_value(instruction->getReturnValue(), bb, vmap);
}

static void
convert_branch_instruction(const llvm::Instruction & i, jive::frontend::basic_block * bb,
	const basic_block_map & bbmap, value_map & vmap)
{
	const llvm::BranchInst * instruction = static_cast<const llvm::BranchInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	JLM_DEBUG_ASSERT(bb->noutedges() == 0);
	for (unsigned n = 0; n < instruction->getNumSuccessors(); n++) {
		JLM_DEBUG_ASSERT(bbmap.find(instruction->getSuccessor(n)) != bbmap.end());
		bb->add_outedge(bbmap.find(instruction->getSuccessor(n))->second, n);
	}
}

static void
convert_switch_instruction(const llvm::Instruction & i, jive::frontend::basic_block * bb,
	const basic_block_map & bbmap, value_map & vmap)
{
	const llvm::SwitchInst * instruction = static_cast<const llvm::SwitchInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	JLM_DEBUG_ASSERT(bb->noutedges() == 0);
	for (unsigned n = 0; n < instruction->getNumSuccessors(); n++) {
		JLM_DEBUG_ASSERT(bbmap.find(instruction->getSuccessor(n)) != bbmap.end());
		bb->add_outedge(bbmap.find(instruction->getSuccessor(n))->second, n);
	}
}

static void
convert_unreachable_instruction(const llvm::Instruction & i, jive::frontend::basic_block * bb,
	const basic_block_map & bbmap, value_map & vmap)
{
	const llvm::UnreachableInst * instruction = static_cast<const llvm::UnreachableInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	JLM_DEBUG_ASSERT(instruction->getNumSuccessors() == 0);
	JLM_DEBUG_ASSERT(bb->noutedges() == 0);

	/* FIXME: this leads to problems:
			- functions don't have a proper return value anymore -> cannot be converted to RVSDG
		, but works for now. We need a proper concept for not returning functions etc.
	*/
	bb->add_outedge(bb->cfg()->exit(), 0);
}

static void
convert_binary_operator(const llvm::Instruction & i, jive::frontend::basic_block * bb,
	const basic_block_map & bbmap, value_map & vmap)
{
	const llvm::BinaryOperator * instruction = static_cast<const llvm::BinaryOperator*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	const jive::frontend::output * result = nullptr;
	switch (instruction->getType()->getTypeID()) {
		case llvm::Type::IntegerTyID:
			result = convert_int_binary_operator(*instruction, bb, vmap);
			break;
		default:
			JLM_DEBUG_ASSERT(0);
	}

	JLM_DEBUG_ASSERT(result);
	vmap[instruction] = result;
}

typedef std::unordered_map<std::type_index, void(*)(const llvm::Instruction&,
	jive::frontend::basic_block*, const basic_block_map&, value_map&)> instruction_map;

static instruction_map imap({
		{std::type_index(typeid(llvm::ReturnInst)), convert_return_instruction}
	, {std::type_index(typeid(llvm::BranchInst)), convert_branch_instruction}
	, {std::type_index(typeid(llvm::SwitchInst)), convert_switch_instruction}
	, {std::type_index(typeid(llvm::UnreachableInst)), convert_unreachable_instruction}
	, {std::type_index(typeid(llvm::BinaryOperator)), convert_binary_operator}
});

void
convert_instruction(const llvm::Instruction & i, jive::frontend::basic_block * bb,
	const basic_block_map & bbmap, value_map & vmap)
{
	/* FIXME: add an JLM_DEBUG_ASSERT here if an instruction is not present */
	if (imap.find(std::type_index(typeid(i))) == imap.end())
		return;

	imap[std::type_index(typeid(i))](i, bb, bbmap, vmap);
}

}

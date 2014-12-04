/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/binops.hpp>
#include <jlm/common.hpp>
#include <jlm/constant.hpp>
#include <jlm/jlm.hpp>
#include <jlm/type.hpp>

#include <jive/frontend/basic_block.h>
#include <jive/frontend/tac/address.h>
#include <jive/frontend/tac/assignment.h>
#include <jive/frontend/tac/bitstring.h>
#include <jive/frontend/tac/phi.h>

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

/* instructions */

static void
convert_return_instruction(const llvm::Instruction & i, jive::frontend::basic_block * bb,
	const basic_block_map & bbmap, value_map & vmap, const jive::frontend::output * state)
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
	const basic_block_map & bbmap, value_map & vmap, const jive::frontend::output * state)
{
	const llvm::BranchInst * instruction = static_cast<const llvm::BranchInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	JLM_DEBUG_ASSERT(bb->noutedges() == 0);
	if (instruction->isConditional()) {
		JLM_DEBUG_ASSERT(instruction->getNumSuccessors() == 2);

		/* taken */
		JLM_DEBUG_ASSERT(bbmap.find(instruction->getSuccessor(0)) != bbmap.end());
		bb->add_outedge(bbmap.find(instruction->getSuccessor(0))->second, 1);

		/* nottaken */
		JLM_DEBUG_ASSERT(bbmap.find(instruction->getSuccessor(1)) != bbmap.end());
		bb->add_outedge(bbmap.find(instruction->getSuccessor(1))->second, 0);
	} else {
		JLM_DEBUG_ASSERT(instruction->isUnconditional());
		JLM_DEBUG_ASSERT(instruction->getNumSuccessors() == 1);
		JLM_DEBUG_ASSERT(bbmap.find(instruction->getSuccessor(0)) != bbmap.end());
		bb->add_outedge(bbmap.find(instruction->getSuccessor(0))->second, 0);
	}
}

static void
convert_switch_instruction(const llvm::Instruction & i, jive::frontend::basic_block * bb,
	const basic_block_map & bbmap, value_map & vmap, const jive::frontend::output * state)
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
	const basic_block_map & bbmap, value_map & vmap, const jive::frontend::output * state)
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
	const basic_block_map & bbmap, value_map & vmap, const jive::frontend::output * state)
{
	const llvm::BinaryOperator * instruction = static_cast<const llvm::BinaryOperator*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	convert_binary_operator(*instruction, bb, vmap, state);
}

static void
convert_comparison_instruction(const llvm::Instruction & i, jive::frontend::basic_block * bb,
	const basic_block_map & bbmap, value_map & vmap, const jive::frontend::output * state)
{
	const llvm::CmpInst * instruction = static_cast<const llvm::CmpInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	convert_comparison_instruction(*instruction, bb, vmap, state);
}

static void
convert_load_instruction(const llvm::Instruction & i, jive::frontend::basic_block * bb,
	const basic_block_map & bbmap, value_map & vmap, const jive::frontend::output * state)
{
	const llvm::LoadInst * instruction = static_cast<const llvm::LoadInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	/* FIXME: handle volatile correctly */

	const jive::frontend::output * address = convert_value(instruction->getPointerOperand(), bb, vmap);
	const jive::frontend::output * result = addrload_tac(bb, address, state,
		*dynamic_cast<jive::value::type*>(convert_type(*instruction->getType()).get()));
	vmap[instruction] = result;
}

static void
convert_store_instruction(const llvm::Instruction & i, jive::frontend::basic_block * bb,
	const basic_block_map & bbmap, value_map & vmap, const jive::frontend::output * state)
{
	const llvm::StoreInst * instruction = static_cast<const llvm::StoreInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	const jive::frontend::output * address = convert_value(instruction->getPointerOperand(), bb, vmap);
	const jive::frontend::output * value = convert_value(instruction->getValueOperand(), bb, vmap);
	const jive::frontend::output * result = addrstore_tac(bb, address, value, state);
	assignment_tac(bb, state, result);
}

static void
convert_phi_instruction(const llvm::Instruction & i, jive::frontend::basic_block * bb,
	const basic_block_map & bbmap, value_map & vmap, const jive::frontend::output * state)
{
	const llvm::PHINode * instruction = static_cast<const llvm::PHINode*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	std::vector<const jive::frontend::output *> ops;
	for (size_t n = 0; n < instruction->getNumIncomingValues(); n++)
		ops.push_back(convert_value(instruction->getIncomingValue(n), bb, vmap));

	const jive::frontend::output * result = phi_tac(bb, ops);
	vmap[instruction] = result;
}

static void
convert_getelementptr_instruction(const llvm::Instruction & i, jive::frontend::basic_block * bb,
	const basic_block_map & bbmap, value_map & vmap, const jive::frontend::output * state)
{
	const llvm::GetElementPtrInst * instruction = static_cast<const llvm::GetElementPtrInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	const jive::frontend::output * base = convert_value(instruction->getPointerOperand(), bb, vmap);
	for (auto idx = instruction->idx_begin(); idx != instruction->idx_end(); idx++) {
		const jive::frontend::output * offset = convert_value(idx->get(), bb, vmap);
		base = addrarraysubscript_tac(bb, base, offset);
	}

	vmap[instruction] = base;
}

typedef std::unordered_map<std::type_index, void(*)(const llvm::Instruction&,
	jive::frontend::basic_block*, const basic_block_map&, value_map&,
	const jive::frontend::output * state)> instruction_map;

static instruction_map imap({
		{std::type_index(typeid(llvm::ReturnInst)), convert_return_instruction}
	, {std::type_index(typeid(llvm::BranchInst)), convert_branch_instruction}
	, {std::type_index(typeid(llvm::SwitchInst)), convert_switch_instruction}
	, {std::type_index(typeid(llvm::UnreachableInst)), convert_unreachable_instruction}
	, {std::type_index(typeid(llvm::BinaryOperator)), convert_binary_operator}
	, {std::type_index(typeid(llvm::ICmpInst)), convert_comparison_instruction}
	, {std::type_index(typeid(llvm::LoadInst)), convert_load_instruction}
	, {std::type_index(typeid(llvm::StoreInst)), convert_store_instruction}
	, {std::type_index(typeid(llvm::PHINode)), convert_phi_instruction}
	, {std::type_index(typeid(llvm::GetElementPtrInst)), convert_getelementptr_instruction}
});

void
convert_instruction(const llvm::Instruction & i, jive::frontend::basic_block * bb,
	const basic_block_map & bbmap, value_map & vmap, const jive::frontend::output * state)
{
	/* FIXME: add an JLM_DEBUG_ASSERT here if an instruction is not present */
	if (imap.find(std::type_index(typeid(i))) == imap.end())
		return;

	imap[std::type_index(typeid(i))](i, bb, bbmap, vmap, state);
}

}

/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/construction/binops.hpp>
#include <jlm/construction/constant.hpp>
#include <jlm/construction/context.hpp>
#include <jlm/construction/jlm.hpp>
#include <jlm/construction/type.hpp>

#include <jlm/IR/address.hpp>
#include <jlm/IR/assignment.hpp>
#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/bitstring.hpp>
#include <jlm/IR/clg.hpp>
#include <jlm/IR/match.hpp>
#include <jlm/IR/select.hpp>

#include <jive/vsdg/controltype.h>

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Function.h>

#include <typeindex>

namespace jlm {

const variable *
convert_value(
	const llvm::Value * v,
	const context & ctx)
{
	if (auto c = dynamic_cast<const llvm::Constant*>(v))
		return convert_constant(*c, ctx.entry_block());

	return ctx.lookup_value(v);
}

/* instructions */

static void
convert_return_instruction(
	const llvm::Instruction & i,
	basic_block * bb,
	const context & ctx)
{
	const llvm::ReturnInst * instruction = static_cast<const llvm::ReturnInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	if (instruction->getReturnValue()) {
		const variable * value = convert_value(instruction->getReturnValue(), ctx);
		assignment_tac(bb, ctx.result(), value);
	}
}

static void
convert_branch_instruction(
	const llvm::Instruction & i,
	basic_block * bb,
	const context & ctx)
{
	const llvm::BranchInst * instruction = static_cast<const llvm::BranchInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	if (instruction->isConditional()) {
		const variable * condition = convert_value(instruction->getCondition(), ctx);
		match_tac(bb, condition, {0});
	}
}

static void
convert_switch_instruction(
	const llvm::Instruction & i,
	basic_block * bb,
	const context & ctx)
{
	const llvm::SwitchInst * instruction = static_cast<const llvm::SwitchInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	JLM_DEBUG_ASSERT(bb->outedges().size() == instruction->getNumCases()+1);
	std::vector<uint64_t> constants(instruction->getNumCases());
	for (auto it = instruction->case_begin(); it != instruction->case_end(); it++) {
		JLM_DEBUG_ASSERT(it != instruction->case_default());
		constants[it.getCaseIndex()] = it.getCaseValue()->getZExtValue();
	}

	const jlm::variable * condition = convert_value(instruction->getCondition(), ctx);
	match_tac(bb, condition, constants);
}

static void
convert_unreachable_instruction(
	const llvm::Instruction & i,
	basic_block * bb,
	const context & ctx)
{
	const llvm::UnreachableInst * instruction = static_cast<const llvm::UnreachableInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);
}

static void
convert_binary_operator(
	const llvm::Instruction & i,
	basic_block * bb,
	const context & ctx)
{
	const llvm::BinaryOperator * instruction = static_cast<const llvm::BinaryOperator*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	convert_binary_operator(*instruction, bb, ctx);
}

static void
convert_comparison_instruction(
	const llvm::Instruction & i,
	basic_block * bb,
	const context & ctx)
{
	const llvm::CmpInst * instruction = static_cast<const llvm::CmpInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	convert_comparison_instruction(*instruction, bb, ctx);
}

static void
convert_load_instruction(
	const llvm::Instruction & i,
	basic_block * bb,
	const context & ctx)
{
	const llvm::LoadInst * instruction = static_cast<const llvm::LoadInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	/* FIXME: handle volatile correctly */

	const variable * address = convert_value(instruction->getPointerOperand(), ctx);
	addrload_tac(bb, address, ctx.state(),
		*dynamic_cast<jive::value::type*>(convert_type(*instruction->getType()).get()),
		ctx.lookup_value(&i));
}

static void
convert_store_instruction(
	const llvm::Instruction & i,
	basic_block * bb,
	const context & ctx)
{
	const llvm::StoreInst * instruction = static_cast<const llvm::StoreInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	const variable * address = convert_value(instruction->getPointerOperand(), ctx);
	const variable * value = convert_value(instruction->getValueOperand(), ctx);
	addrstore_tac(bb, address, value, ctx.state(), ctx.state());
}

static void
convert_phi_instruction(
	const llvm::Instruction & i,
	basic_block * bb,
	const context & ctx)
{
	const llvm::PHINode * phi = static_cast<const llvm::PHINode*>(&i);
	JLM_DEBUG_ASSERT(phi != nullptr);

	std::vector<const jlm::variable*> operands;
	for (auto edge : bb->inedges()) {
		const basic_block * tmp = static_cast<basic_block*>(edge->source());
		const llvm::BasicBlock * ib = ctx.lookup_basic_block(tmp);
		const jlm::variable * v = convert_value(phi->getIncomingValueForBlock(ib), ctx);
		operands.push_back(v);
	}

	JLM_DEBUG_ASSERT(operands.size() != 0);
	bb->append(phi_op(operands.size(), operands[0]->type()), operands, {ctx.lookup_value(phi)});
}

static void
convert_getelementptr_instruction(
	const llvm::Instruction & i,
	basic_block * bb,
	const context & ctx)
{
	const llvm::GetElementPtrInst * instruction = static_cast<const llvm::GetElementPtrInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	const jlm::variable * base = convert_value(instruction->getPointerOperand(), ctx);
	for (auto idx = instruction->idx_begin(); idx != instruction->idx_end(); idx++) {
		const jlm::variable * offset = convert_value(idx->get(), ctx);
		base = addrarraysubscript_tac(bb, base, offset, ctx.lookup_value(&i));
	}
}

static void
convert_trunc_instruction(
	const llvm::Instruction & i,
	basic_block * bb,
	const context & ctx)
{
	const llvm::TruncInst * instruction = static_cast<const llvm::TruncInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	const llvm::IntegerType * dst_type = static_cast<const llvm::IntegerType*>(i.getType());
	const jlm::variable * operand = convert_value(instruction->getOperand(0), ctx);
	bitslice_tac(bb, operand, 0, dst_type->getBitWidth(), ctx.lookup_value(&i));
}

static void
convert_call_instruction(
	const llvm::Instruction & i,
	basic_block * bb,
	const context & ctx)
{
	const llvm::CallInst * instruction = static_cast<const llvm::CallInst*>(&i);
	JLM_DEBUG_ASSERT(instruction != nullptr);

	llvm::Function * f = instruction->getCalledFunction();

	jlm::clg_node * caller = bb->cfg()->function();
	jlm::clg_node * callee = caller->clg().lookup_function(f->getName());
	JLM_DEBUG_ASSERT(callee != nullptr);
	caller->add_call(callee);

	std::vector<const jlm::variable*> arguments;
	for (size_t n = 0; n < instruction->getNumArgOperands(); n++)
		arguments.push_back(convert_value(instruction->getArgOperand(n), ctx));
	arguments.push_back(ctx.state());

	jive::fct::type type = dynamic_cast<jive::fct::type&>(*convert_type(*f->getFunctionType()).get());

	std::vector<const jlm::variable*> results;
	if (type.nreturns() == 2)
		results.push_back(ctx.lookup_value(&i));
	results.push_back(ctx.state());

	bb->append(jlm::apply_op(callee), arguments, results);
}

static void
convert_select_instruction(
	const llvm::Instruction & i,
	basic_block * bb,
	const context & ctx)
{
	JLM_DEBUG_ASSERT(dynamic_cast<const llvm::SelectInst*>(&i));
	const llvm::SelectInst * instruction = static_cast<const llvm::SelectInst*>(&i);

	const jlm::variable * condition = convert_value(instruction->getCondition(), ctx);
	const jlm::variable * tv = convert_value(instruction->getTrueValue(), ctx);
	const jlm::variable * fv = convert_value(instruction->getFalseValue(), ctx);
	select_tac(bb, condition, tv, fv, ctx.lookup_value(instruction));
}

typedef std::unordered_map<
		std::type_index,
		void(*)(const llvm::Instruction&, jlm::basic_block*, const context&)
	> instruction_map;

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
	, {std::type_index(typeid(llvm::TruncInst)), convert_trunc_instruction}
	, {std::type_index(typeid(llvm::CallInst)), convert_call_instruction}
	, {std::type_index(typeid(llvm::SelectInst)), convert_select_instruction}
});

void
convert_instruction(
	const llvm::Instruction & i,
	basic_block * bb,
	const context & ctx)
{
	/* FIXME: add an JLM_DEBUG_ASSERT here if an instruction is not present */
	if (imap.find(std::type_index(typeid(i))) == imap.end())
		return;

	imap[std::type_index(typeid(i))](i, bb, ctx);
}

}

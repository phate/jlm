/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/construction/constant.hpp>
#include <jlm/construction/context.hpp>
#include <jlm/construction/instruction.hpp>
#include <jlm/construction/module.hpp>
#include <jlm/construction/type.hpp>
#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/cfg.hpp>
#include <jlm/IR/cfg_node.hpp>
#include <jlm/IR/clg.hpp>
#include <jlm/IR/module.hpp>
#include <jlm/IR/operators.hpp>

#include <jive/arch/memorytype.h>
#include <jive/vsdg/basetype.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace jlm
{


static cfg_node *
create_cfg_structure(
	const llvm::Function & function,
	cfg * cfg,
	basic_block_map & bbmap)
{
	basic_block * entry_block = cfg->create_basic_block();
	cfg->exit()->divert_inedges(entry_block);

	/* create all basic_blocks */
	auto it = function.getBasicBlockList().begin();
	for (; it != function.getBasicBlockList().end(); it++)
			bbmap.insert_basic_block(&(*it), cfg->create_basic_block());

	entry_block->add_outedge(bbmap[&function.getEntryBlock()], 0);

	/* create CFG structure */
	it = function.getBasicBlockList().begin();
	for (; it != function.getBasicBlockList().end(); it++) {
		const llvm::TerminatorInst * instr = it->getTerminator();
		if (dynamic_cast<const llvm::ReturnInst*>(instr)) {
			bbmap[&(*it)]->add_outedge(cfg->exit(), 0);
			continue;
		}

		if (dynamic_cast<const llvm::UnreachableInst*>(instr)) {
			bbmap[&(*it)]->add_outedge(cfg->exit(), 0);
			continue;
		}

		if (auto branch = dynamic_cast<const llvm::BranchInst*>(instr)) {
			if (branch->isConditional()) {
				JLM_DEBUG_ASSERT(branch->getNumSuccessors() == 2);
				bbmap[&(*it)]->add_outedge(bbmap[branch->getSuccessor(0)], 1);
				bbmap[&(*it)]->add_outedge(bbmap[branch->getSuccessor(1)], 0);
				continue;
			}
		}

		if (auto swi = dynamic_cast<const llvm::SwitchInst*>(instr)) {
			for (auto c = swi->case_begin(); c != swi->case_end(); c++) {
				JLM_DEBUG_ASSERT(c != swi->case_default());
				bbmap[&(*it)]->add_outedge(bbmap[c.getCaseSuccessor()], c.getCaseIndex());
			}
			bbmap[&(*it)]->add_outedge(bbmap[swi->case_default().getCaseSuccessor()], swi->getNumCases());
			continue;
		}

		for (size_t n = 0; n < instr->getNumSuccessors(); n++)
			bbmap[&(*it)]->add_outedge(bbmap[instr->getSuccessor(n)], n);
	}

	if (cfg->exit()->ninedges() > 1) {
		jlm::basic_block * bb = cfg->create_basic_block();
		cfg->exit()->divert_inedges(bb);
		bb->add_outedge(cfg->exit(), 0);
	}

	return entry_block;
}

static void
convert_basic_block(
	const llvm::BasicBlock & bb,
	context & ctx)
{
	/* forward declare all instructions, except terminator instructions */
	for (auto it = bb.begin(); it != bb.end(); it++) {
		if (dynamic_cast<const llvm::TerminatorInst*>(&(*it)))
			continue;

		jlm::cfg * cfg = ctx.entry_block()->cfg();
		if (it->getType()->getTypeID() != llvm::Type::VoidTyID)
			ctx.insert_value(&(*it), cfg->create_variable(*convert_type(it->getType(), ctx)));
	}

	for (auto it = bb.begin(); it != bb.end(); it++)
		convert_instruction(&(*it), ctx.lookup_basic_block(&bb), ctx);
}

static void
convert_function(
	const llvm::Function & function,
	jlm::clg_node * clg_node,
	context & ctx)
{
	if (function.isDeclaration())
		return;

	std::vector<std::string> names;
	llvm::Function::ArgumentListType::const_iterator jt = function.getArgumentList().begin();
	for (; jt != function.getArgumentList().end(); jt++)
		names.push_back(jt->getName().str());
	names.push_back("_s_");

	std::vector<const jlm::variable*> arguments = clg_node->cfg_begin(names);
	const jlm::variable * state = arguments.back();
	jlm::cfg * cfg = clg_node->cfg();

	basic_block_map bbmap;
	basic_block * entry_block;
	entry_block = static_cast<basic_block*>(create_cfg_structure(function, cfg, bbmap));

	const jlm::variable * result = nullptr;
	if (!function.getReturnType()->isVoidTy())
		result = cfg->create_variable(*convert_type(function.getReturnType(), ctx), "_r_");

	ctx.set_basic_block_map(bbmap);
	ctx.set_entry_block(entry_block);
	ctx.set_state(state);
	ctx.set_result(result);
	if (!function.getReturnType()->isVoidTy()) {
		const variable * udef = entry_block->append(*create_undef_value(function.getReturnType(), ctx));
		entry_block->append(assignment_op(result->type()), {udef}, {result});
	}

	jt = function.getArgumentList().begin();
	for (size_t n = 0; jt != function.getArgumentList().end(); jt++, n++)
		ctx.insert_value(&(*jt), arguments[n]);

	auto it = function.getBasicBlockList().begin();
	for (; it != function.getBasicBlockList().end(); it++)
		convert_basic_block(*it, ctx);

	std::vector<const jlm::variable*> results;
	if (function.getReturnType()->getTypeID() != llvm::Type::VoidTyID)
		results.push_back(result);
	results.push_back(state);

	clg_node->cfg_end(results);
	clg_node->cfg()->prune();
}

static void
convert_functions(
	const llvm::Module::FunctionListType & list,
	jlm::clg & clg,
	context & ctx)
{
	for (auto it = list.begin(); it != list.end(); it++) {
		jive::fct::type fcttype(dynamic_cast<const jive::fct::type&>(
			*convert_type((*it).getFunctionType(), ctx)));
		clg.add_function((*it).getName().str().c_str(), fcttype);
	}

	for (auto it = list.begin(); it != list.end(); it++)
		convert_function(*it, clg.lookup_function((*it).getName().str()), ctx);
}

static void
convert_global_variables(
	const llvm::Module::GlobalListType & list,
	jlm::module & mod,
	context & ctx)
{
	for (auto it = list.begin(); it != list.end(); it++) {
		const variable * v = mod.add_global_variable(it->getName().str(),
			*convert_constant(it.getNodePtrUnchecked(), ctx));
		ctx.insert_value(it.getNodePtrUnchecked(), v);
	}
}

void
convert_module(const llvm::Module & module, jlm::module & mod)
{
	context ctx;
	convert_global_variables(module.getGlobalList(), mod, ctx);
	convert_functions(module.getFunctionList(), mod.clg(), ctx);
}

}

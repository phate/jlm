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
	auto entry_block = create_basic_block_node(cfg);
	cfg->exit()->divert_inedges(entry_block);

	/* create all basic_blocks */
	auto it = function.getBasicBlockList().begin();
	for (; it != function.getBasicBlockList().end(); it++)
			bbmap.insert_basic_block(&(*it), create_basic_block_node(cfg));

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
				auto bb = create_basic_block_node(cfg);
				bbmap[&(*it)]->add_outedge(bb, c.getCaseIndex());
				bb->add_outedge(bbmap[c.getCaseSuccessor()], 0);
			}
			auto bb = create_basic_block_node(cfg);
			bbmap[&(*it)]->add_outedge(bb, swi->getNumCases());
			bb->add_outedge(bbmap[swi->case_default().getCaseSuccessor()], 0);
			continue;
		}

		for (size_t n = 0; n < instr->getNumSuccessors(); n++)
			bbmap[&(*it)]->add_outedge(bbmap[instr->getSuccessor(n)], n);
	}

	if (cfg->exit()->ninedges() > 1) {
		auto bb = create_basic_block_node(cfg);
		cfg->exit()->divert_inedges(bb);
		bb->add_outedge(cfg->exit(), 0);
	}

	return entry_block;
}

static void
convert_basic_blocks(
	const llvm::Function::BasicBlockListType & bbs,
	context & ctx)
{
	/* forward declare all instructions, except terminator instructions */
	for (auto bb = bbs.begin(); bb != bbs.end(); bb++) {
		for (auto i = bb->begin(); i != bb->end(); i++) {
			if (dynamic_cast<const llvm::TerminatorInst*>(&(*i)))
				continue;

			if (i->getType()->getTypeID() != llvm::Type::VoidTyID)
				ctx.insert_value(&(*i), create_variable(*convert_type(i->getType(), ctx)));
		}
	}

	for (auto bb = bbs.begin(); bb != bbs.end(); bb++) {
		for (auto i = bb->begin(); i != bb->end(); i++)
			convert_instruction(&(*i), ctx.lookup_basic_block(&(*bb)), ctx);
	}
}

static void
convert_function(
	const llvm::Function & function,
	context & ctx)
{
	if (function.isDeclaration())
		return;

	auto clg_node = static_cast<jlm::clg_node*>(ctx.lookup_value(&function).get());

	std::vector<std::string> names;
	llvm::Function::ArgumentListType::const_iterator jt = function.getArgumentList().begin();
	for (; jt != function.getArgumentList().end(); jt++)
		names.push_back(jt->getName().str());
	if (function.isVarArg())
		names.push_back("_varg_");
	names.push_back("_s_");

	auto arguments = clg_node->cfg_begin(names);
	auto state = arguments.back();
	jlm::cfg * cfg = clg_node->cfg();

	basic_block_map bbmap;
	auto entry_block = create_cfg_structure(function, cfg, bbmap);

	std::shared_ptr<const variable> result = nullptr;
	if (!function.getReturnType()->isVoidTy())
		result = create_variable(*convert_type(function.getReturnType(), ctx), "_r_");

	ctx.set_basic_block_map(bbmap);
	ctx.set_entry_block(entry_block);
	ctx.set_state(state);
	ctx.set_result(result);
	if (!function.getReturnType()->isVoidTy()) {
		auto attr = static_cast<basic_block*>(&entry_block->attribute());
		auto udef = attr->append(ctx.cfg(), *create_undef_value(function.getReturnType(), ctx));
		attr->append(create_assignment(result->type(), {udef}, {result}));
	}

	jt = function.getArgumentList().begin();
	for (size_t n = 0; jt != function.getArgumentList().end(); jt++, n++)
		ctx.insert_value(&(*jt), arguments[n]);

	convert_basic_blocks(function.getBasicBlockList(), ctx);

	std::vector<std::shared_ptr<const jlm::variable>> results;
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
		/*clg_node * f = */clg.add_function(
			(*it).getName().str().c_str(),
			fcttype,
			it->getLinkage() != llvm::GlobalValue::InternalLinkage);
		/* FIXME */
		ctx.insert_value(&(*it), nullptr);
	}

	for (auto it = list.begin(); it != list.end(); it++)
		convert_function(*it, ctx);
}

static void
convert_global_variables(
	const llvm::Module::GlobalListType & list,
	jlm::module & mod,
	context & ctx)
{
	for (auto it = list.begin(); it != list.end(); it++) {
	/* FIXME
		variable * v = mod.add_global_variable(
			it->getName().str(),
			*convert_constant(it.getNodePtrUnchecked(), ctx),
			it->getLinkage() != llvm::GlobalValue::InternalLinkage);
		ctx.insert_value(it.getNodePtrUnchecked(), v);
	*/
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

/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/llvm2jlm/constant.hpp>
#include <jlm/llvm2jlm/context.hpp>
#include <jlm/llvm2jlm/instruction.hpp>
#include <jlm/llvm2jlm/module.hpp>
#include <jlm/llvm2jlm/type.hpp>
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
	cfg->exit_node()->divert_inedges(entry_block);

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
			bbmap[&(*it)]->add_outedge(cfg->exit_node(), 0);
			continue;
		}

		if (dynamic_cast<const llvm::UnreachableInst*>(instr)) {
			bbmap[&(*it)]->add_outedge(cfg->exit_node(), 0);
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

	if (cfg->exit_node()->ninedges() > 1) {
		auto bb = create_basic_block_node(cfg);
		cfg->exit_node()->divert_inedges(bb);
		bb->add_outedge(cfg->exit_node(), 0);
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

			if (i->getType()->getTypeID() != llvm::Type::VoidTyID) {
				auto v = ctx.module().create_variable(*convert_type(i->getType(), ctx), false);
				ctx.insert_value(&(*i), v);
			}
		}
	}

	for (auto bb = bbs.begin(); bb != bbs.end(); bb++) {
		for (auto i = bb->begin(); i != bb->end(); i++)
			convert_instruction(&(*i), ctx.lookup_basic_block(&(*bb)), ctx);
	}
}

std::unique_ptr<jlm::cfg>
create_cfg(const llvm::Function & f, context & ctx)
{
	auto node = static_cast<const function_variable*>(ctx.lookup_value(&f))->function();

	std::unique_ptr<jlm::cfg> cfg(new jlm::cfg(ctx.module()));

	/* add arguments */
	size_t n = 0;
	for (const auto & arg : f.getArgumentList()) {
		JLM_DEBUG_ASSERT(n < node->type().narguments());
		auto & type = node->type().argument_type(n++);
		auto v = ctx.module().create_variable(type, arg.getName().str(), false);
		cfg->entry().append_argument(v);
		ctx.insert_value(&arg, v);
	}
	if (f.isVarArg()) {
		JLM_DEBUG_ASSERT(n < node->type().narguments());
		auto v = ctx.module().create_variable(node->type().argument_type(n++), "_varg_", false);
		cfg->entry().append_argument(v);
	}
	JLM_DEBUG_ASSERT(n < node->type().narguments());
	auto state = ctx.module().create_variable(node->type().argument_type(n++), "_s_", false);
	cfg->entry().append_argument(state);
	JLM_DEBUG_ASSERT(n == node->type().narguments());

	/* create cfg structure */
	basic_block_map bbmap;
	auto entry_block = create_cfg_structure(f, cfg.get(), bbmap);

	/* add results */
	const variable * result = nullptr;
	if (!f.getReturnType()->isVoidTy()) {
		result = ctx.module().create_variable(*convert_type(f.getReturnType(), ctx), "_r_", false);
		auto attr = static_cast<basic_block*>(&entry_block->attribute());
		auto e = create_undef_value(f.getReturnType(), ctx);
		auto tacs = expr2tacs(*e, ctx);
		attr->append(tacs);
		attr->append(create_assignment(result->type(), {attr->last()->output(0)}, {result}));

		JLM_DEBUG_ASSERT(node->type().nresults() == 2);
		JLM_DEBUG_ASSERT(result->type() == node->type().result_type(0));
		cfg->exit().append_result(result);
	}
	cfg->exit().append_result(state);

	/* convert instructions */
	ctx.set_basic_block_map(bbmap);
	ctx.set_entry_block(entry_block);
	ctx.set_state(state);
	ctx.set_result(result);
	convert_basic_blocks(f.getBasicBlockList(), ctx);

	cfg->prune();
	return cfg;
}

static void
convert_function(
	const llvm::Function & function,
	context & ctx)
{
	if (function.isDeclaration())
		return;

	auto fv = ctx.lookup_value(&function);
	auto node = static_cast<const function_variable*>(fv)->function();

	node->add_cfg(create_cfg(function, ctx));
}

static void
convert_functions(
	const llvm::Module::FunctionListType & list,
	jlm::clg & clg,
	context & ctx)
{
	for (const auto & f : list) {
		jive::fct::type fcttype(dynamic_cast<const jive::fct::type&>(
			*convert_type(f.getFunctionType(), ctx)));
		clg_node * n = clg.add_function(
			f.getName().str().c_str(),
			fcttype,
			f.getLinkage() != llvm::GlobalValue::InternalLinkage);
		ctx.insert_value(&f, ctx.module().create_variable(n));
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

std::unique_ptr<module>
convert_module(const llvm::Module & module)
{
	std::unique_ptr<jlm::module> m(new jlm::module());

	context ctx(*m);
	convert_global_variables(module.getGlobalList(), *m, ctx);
	convert_functions(module.getFunctionList(), m->clg(), ctx);

	return m;
}

}

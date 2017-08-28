/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/basic_block.hpp>
#include <jlm/ir/cfg.hpp>
#include <jlm/ir/cfg-structure.hpp>
#include <jlm/ir/cfg_node.hpp>
#include <jlm/ir/clg.hpp>
#include <jlm/ir/module.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/llvm2jlm/constant.hpp>
#include <jlm/llvm2jlm/context.hpp>
#include <jlm/llvm2jlm/instruction.hpp>
#include <jlm/llvm2jlm/module.hpp>
#include <jlm/llvm2jlm/type.hpp>

#include <jive/arch/memorytype.h>
#include <jive/vsdg/basetype.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace jlm
{

static void
convert_basic_blocks(
	llvm::Function::BasicBlockListType & bbs,
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

	for (auto & bb : bbs) {
		for (auto & instruction : bb) {
			std::vector<std::unique_ptr<jlm::tac>> tacs;
			convert_instruction(&instruction, tacs, ctx);
			append_last(ctx.lookup_basic_block(&bb), tacs);
		}
	}
}

std::unique_ptr<jlm::cfg>
create_cfg(llvm::Function & f, context & ctx)
{
	auto node = static_cast<const fctvariable*>(ctx.lookup_value(&f))->function();
	auto & m = ctx.module();

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
		auto v = m.create_variable(node->type().argument_type(n++), "_varg_", false);
		cfg->entry().append_argument(v);
	}
	JLM_DEBUG_ASSERT(n < node->type().narguments());
	auto state = m.create_variable(node->type().argument_type(n++), "_s_", false);
	cfg->entry().append_argument(state);
	JLM_DEBUG_ASSERT(n == node->type().narguments());

	/* create all basic blocks */
	basic_block_map bbmap;
	for (const auto & bb : f.getBasicBlockList())
			bbmap.insert_basic_block(&bb, create_basic_block_node(cfg.get()));

	/* create entry block */
	auto entry_block = create_basic_block_node(cfg.get());
	cfg->exit_node()->divert_inedges(entry_block);
	entry_block->add_outedge(bbmap[&f.getEntryBlock()]);

	/* add results */
	jlm::variable * result = nullptr;
	if (!f.getReturnType()->isVoidTy()) {
		result = m.create_variable(*convert_type(f.getReturnType(), ctx), "_r_", false);
		append_last(entry_block, create_undef_constant_tac(result));

		JLM_DEBUG_ASSERT(node->type().nresults() == 2);
		JLM_DEBUG_ASSERT(result->type() == node->type().result_type(0));
		cfg->exit().append_result(result);
	}
	cfg->exit().append_result(state);

	/* convert instructions */
	ctx.set_basic_block_map(bbmap);
	ctx.set_state(state);
	ctx.set_result(result);
	convert_basic_blocks(f.getBasicBlockList(), ctx);

	/* ensure that exit node has only one incoming edge */
	if (cfg->exit_node()->ninedges() > 1) {
		auto bb = create_basic_block_node(cfg.get());
		cfg->exit_node()->divert_inedges(bb);
		bb->add_outedge(cfg->exit_node());
	}

	straighten(*cfg);
	prune(*cfg);
	return cfg;
}

static void
convert_function(llvm::Function & function, context & ctx)
{
	if (function.isDeclaration())
		return;

	auto fv = ctx.lookup_value(&function);
	auto node = static_cast<const fctvariable*>(fv)->function();

	node->add_cfg(create_cfg(function, ctx));
}

static const jlm::linkage &
convert_linkage(const llvm::GlobalValue::LinkageTypes & linkage)
{
	static std::unordered_map<llvm::GlobalValue::LinkageTypes, jlm::linkage> map({
	  {llvm::GlobalValue::ExternalLinkage, jlm::linkage::external_linkage}
	, {llvm::GlobalValue::AvailableExternallyLinkage, jlm::linkage::available_externally_linkage}
	, {llvm::GlobalValue::LinkOnceAnyLinkage, jlm::linkage::link_once_any_linkage}
	, {llvm::GlobalValue::LinkOnceODRLinkage, jlm::linkage::link_once_odr_linkage}
	, {llvm::GlobalValue::WeakAnyLinkage, jlm::linkage::weak_any_linkage}
	, {llvm::GlobalValue::WeakODRLinkage, jlm::linkage::weak_odr_linkage}
	, {llvm::GlobalValue::AppendingLinkage, jlm::linkage::appending_linkage}
	, {llvm::GlobalValue::InternalLinkage, jlm::linkage::internal_linkage}
	, {llvm::GlobalValue::PrivateLinkage, jlm::linkage::private_linkage}
	, {llvm::GlobalValue::ExternalWeakLinkage, jlm::linkage::external_weak_linkage}
	, {llvm::GlobalValue::CommonLinkage, jlm::linkage::common_linkage}
	});

	JIVE_DEBUG_ASSERT(map.find(linkage) != map.end());
	return map[linkage];
}

static void
convert_functions(
	llvm::Module::FunctionListType & list,
	jlm::clg & clg,
	context & ctx)
{
	for (const auto & f : list) {
		jive::fct::type fcttype(dynamic_cast<const jive::fct::type&>(
			*convert_type(f.getFunctionType(), ctx)));
		auto n = clg_node::create(
			clg,
			f.getName().str(),
			fcttype,
			f.getLinkage() != llvm::GlobalValue::InternalLinkage);
		ctx.insert_value(&f, ctx.module().create_variable(n, convert_linkage(f.getLinkage())));
	}

	for (auto it = list.begin(); it != list.end(); it++)
		convert_function(*it, ctx);
}

static void
convert_global_variables(llvm::Module::GlobalListType & vs, context & ctx)
{
	auto & m = ctx.module();

	for (auto & gv : vs) {
		std::unique_ptr<const expr> init;
		if (gv.hasInitializer()) init = convert_constant_expression(&gv, ctx);

		auto name = gv.getName().str();
		auto type = convert_type(gv.getType(), ctx);
		auto linkage = convert_linkage(gv.getLinkage());
		auto v = m.create_global_value(*type, name, linkage, std::move(init));
		ctx.insert_value(&gv, v);
	}
}

std::unique_ptr<module>
convert_module(llvm::Module & module)
{
	auto tt = module.getTargetTriple();
	auto dl = module.getDataLayoutStr();
	std::unique_ptr<jlm::module> m(new jlm::module(tt, dl));

	context ctx(*m);
	convert_global_variables(module.getGlobalList(), ctx);
	convert_functions(module.getFunctionList(), m->clg(), ctx);

	return m;
}

}

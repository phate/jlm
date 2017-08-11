/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/arch/memorytype.h>
#include <jive/vsdg/operators/match.h>

#include <jlm/ir/basic_block.hpp>
#include <jlm/ir/cfg.hpp>
#include <jlm/ir/cfg-structure.hpp>
#include <jlm/ir/cfg_node.hpp>
#include <jlm/ir/module.hpp>
#include <jlm/ir/operators.hpp>

#include <jlm/jlm2llvm/context.hpp>
#include <jlm/jlm2llvm/instruction.hpp>
#include <jlm/jlm2llvm/jlm2llvm.hpp>
#include <jlm/jlm2llvm/type.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <deque>
#include <unordered_map>

static inline std::vector<jlm::cfg_node*>
breadth_first_traversal(const jlm::cfg & cfg)
{
	std::deque<jlm::cfg_node*> next({cfg.entry_node()});
	std::vector<jlm::cfg_node*> nodes({cfg.entry_node()});
	std::unordered_set<jlm::cfg_node*> visited({cfg.entry_node()});
	while (!next.empty()) {
		auto node = next.front();
		next.pop_front();

		for (auto it = node->begin_outedges(); it != node->end_outedges(); it++) {
			if (visited.find(it->sink()) == visited.end()) {
				visited.insert(it->sink());
				next.push_back(it->sink());
				nodes.push_back(it->sink());
			}
		}
	}

	return nodes;
}

static inline llvm::GlobalValue::LinkageTypes
function_linkage(const jlm::clg_node & f, const jlm::module & module)
{
	JLM_DEBUG_ASSERT(module.variable(&f));

	if (!f.cfg())
		return llvm::GlobalValue::ExternalLinkage;

	if (module.variable(&f)->exported())
		return llvm::GlobalValue::ExternalLinkage;

	return llvm::GlobalValue::InternalLinkage;
}

namespace jlm {
namespace jlm2llvm {

static inline const jlm::tac *
find_match_tac(const jlm::basic_block & bb)
{
	auto it = bb.rbegin();
	const jlm::tac * tac = nullptr;
	while (it != bb.rend()) {
		if (*it && dynamic_cast<const jive::match_op*>(&(*it)->operation())) {
			tac = *it;
			break;
		}

		it = std::next(it);
	}

	return tac;
}

static inline void
create_terminator_instruction(const jlm::cfg_node * node, context & ctx)
{
	JLM_DEBUG_ASSERT(is_basic_block(node->attribute()));
	auto & bb = *static_cast<const jlm::basic_block*>(&node->attribute());
	auto & lctx = ctx.llvm_module().getContext();
	auto & cfg = *node->cfg();

	llvm::IRBuilder<> builder(ctx.basic_block(node));

	/* unconditional branch or return statement */
	if (node->noutedges() == 1) {
		auto target = node->outedge(0)->sink();

		/* unconditional branch */
		if (target != cfg.exit_node()) {
			builder.CreateBr(ctx.basic_block(target));
			return;
		}

		/* return without result */
		if (cfg.exit().nresults() == 1) {
			builder.CreateRetVoid();
			return;
		}

		/* return with result */
		builder.CreateRet(ctx.value(cfg.exit().result(0)));
		return;
	}

	/* conditional branch */
	if (node->noutedges() == 2) {
		JLM_DEBUG_ASSERT(node->outedge(0)->sink() != cfg.exit_node());
		JLM_DEBUG_ASSERT(node->outedge(1)->sink() != cfg.exit_node());

		const auto & branch = bb.last();
		JLM_DEBUG_ASSERT(branch && dynamic_cast<const jlm::branch_op*>(&branch->operation()));
		auto condition = ctx.value(branch->input(0));
		auto bbtrue = ctx.basic_block(node->outedge(0)->sink());
		auto bbfalse = ctx.basic_block(node->outedge(1)->sink());
		builder.CreateCondBr(condition, bbtrue, bbfalse);
		return;
	}

	/* switch */
	const auto & branch = bb.last();
	JLM_DEBUG_ASSERT(branch && is_branch_op(branch->operation()));
	auto condition = ctx.value(branch->input(0));
	auto match = find_match_tac(bb);
	JLM_DEBUG_ASSERT(match && match->output(0) == branch->input(0));
	auto mop = static_cast<const jive::match_op*>(&match->operation());

	auto defbb = ctx.basic_block(node->outedge(mop->default_alternative())->sink());
	auto sw = builder.CreateSwitch(condition, defbb);
	for (const auto & alt : *mop) {
		auto & type = *static_cast<const jive::bits::type*>(&mop->argument_type(0));
		auto value = llvm::ConstantInt::get(convert_type(type, lctx), alt.first);
		sw->addCase(value, ctx.basic_block(node->outedge(alt.second)->sink()));
	}
}

static inline void
convert_cfg(jlm::cfg & cfg, llvm::Function & f, context & ctx)
{
	JLM_DEBUG_ASSERT(is_closed(cfg));

	straighten(cfg);
	auto nodes = breadth_first_traversal(cfg);

	/* create basic blocks */
	for (const auto & node : nodes) {
		if (node == cfg.entry_node() || node == cfg.exit_node())
			continue;

		auto bb = llvm::BasicBlock::Create(f.getContext(), strfmt("bb", &node), &f);
		ctx.insert(node, bb);
	}

	/* add arguments to context */
	size_t n = 0;
	for (auto & arg : f.getArgumentList())
		ctx.insert(cfg.entry().argument(n++), &arg);

	/* create non-terminator instructions */
	for (const auto & node : nodes) {
		if (node == cfg.entry_node() || node == cfg.exit_node())
			continue;

		JLM_DEBUG_ASSERT(is_basic_block(node->attribute()));
		auto & bb = *static_cast<const jlm::basic_block*>(&node->attribute());
		for (const auto & tac : bb)
			convert_instruction(*tac, node, ctx);
	}

	/* create cfg structure */
	for (const auto & node : nodes) {
		if (node == cfg.entry_node() || node == cfg.exit_node())
			continue;

		create_terminator_instruction(node, ctx);
	}

	/* patch phi instructions */
	for (const auto & node : nodes) {
		if (node == cfg.entry_node() || node == cfg.exit_node())
			continue;

		JLM_DEBUG_ASSERT(is_basic_block(node->attribute()));
		auto & bb = *static_cast<const jlm::basic_block*>(&node->attribute());
		for (const auto & tac : bb) {
			if (!is_phi_op(tac->operation()))
				continue;
			if (dynamic_cast<const jive::mem::type*>(&tac->output(0)->type()))
				continue;

			JLM_DEBUG_ASSERT(node->ninedges() == tac->ninputs());
			auto & op = *static_cast<const jlm::phi_op*>(&tac->operation());
			auto phi = llvm::dyn_cast<llvm::PHINode>(ctx.value(tac->output(0)));
			for (size_t n = 0; n < tac->ninputs(); n++)
				phi->addIncoming(ctx.value(tac->input(n)), ctx.basic_block(op.node(n)));
		}
	}
}

static inline void
convert_function(const jlm::clg_node & node, context & ctx)
{
	if (!node.cfg())
		return;

	auto & jm = ctx.jlm_module();
	auto f = llvm::cast<llvm::Function>(ctx.value(jm.variable(&node)));
	convert_cfg(*node.cfg(), *f, ctx);
}

static inline void
convert_callgraph(const jlm::clg & clg, context & ctx)
{
	auto & jm = ctx.jlm_module();
	auto & lm = ctx.llvm_module();

	/* forward declare all functions */
	for (const auto & node : jm.clg().nodes()) {
		auto ftype = convert_type(node->type(), lm.getContext());
		auto f = llvm::Function::Create(ftype, function_linkage(*node, jm), node->name(), &lm);
		ctx.insert(jm.variable(node), f);
	}

	/* convert all functions */
	for (const auto & node : jm.clg().nodes())
		convert_function(*node, ctx);
}

static inline void
convert_globals(context & ctx)
{
	using namespace llvm;

	auto & jm = ctx.jlm_module();
	auto & lm = ctx.llvm_module();

	for (const auto & gv : jm) {
		auto variable = gv.first;
		auto value = convert_expression(*gv.second, ctx);

		JLM_DEBUG_ASSERT(is_ptrtype(variable->type()));
		auto pt = static_cast<const jlm::ptrtype*>(&variable->type());
		auto type = convert_type(pt->pointee_type(), lm.getContext());
		auto linkage = GlobalValue::InternalLinkage;
		if (variable->exported()) linkage = GlobalValue::ExternalLinkage;

		/* FIXME: isConstant parameter is not always true */
		auto addr = new GlobalVariable(lm, type, true, linkage, value, variable->name());
		ctx.insert(variable, addr);
	}
}

std::unique_ptr<llvm::Module>
convert(jlm::module & jm, llvm::LLVMContext & lctx)
{
	std::unique_ptr<llvm::Module> lm(new llvm::Module("module", lctx));
	lm->setTargetTriple(jm.target_triple());

	context ctx(jm, *lm);
	convert_globals(ctx);
	convert_callgraph(jm.clg(), ctx);

	return lm;
}

}}

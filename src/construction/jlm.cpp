/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/construction/constant.hpp>
#include <jlm/construction/context.hpp>
#include <jlm/construction/instruction.hpp>
#include <jlm/construction/jlm.hpp>
#include <jlm/construction/type.hpp>
#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/cfg.hpp>
#include <jlm/IR/cfg_node.hpp>
#include <jlm/IR/tac/assignment.hpp>

#include <jive/arch/memorytype.h>
#include <jive/vsdg/basetype.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace jlm
{

typedef std::unordered_map<const llvm::Function*, jlm::frontend::clg_node*> function_map;

static frontend::cfg_node *
create_cfg_structure(
	const llvm::Function & function,
	frontend::cfg * cfg,
	basic_block_map & bbmap)
{
	frontend::basic_block * entry_block = cfg->create_basic_block();
	cfg->exit()->divert_inedges(entry_block);

	/* create all basic_blocks */
	auto it = function.getBasicBlockList().begin();
	for (; it != function.getBasicBlockList().end(); it++)
			bbmap.insert_basic_block(&(*it), cfg->create_basic_block());

	entry_block->add_outedge(bbmap[&function.getEntryBlock()], 0);

	/* create CFG structure */
	it = function.getBasicBlockList().begin();
	for (; it != function.getBasicBlockList().end(); it++) {
		if (dynamic_cast<const llvm::ReturnInst*>(it->getTerminator())) {
			bbmap[&(*it)]->add_outedge(cfg->exit(), 0);
			continue;
		}

		if (dynamic_cast<const llvm::UnreachableInst*>(it->getTerminator())) {
			bbmap[&(*it)]->add_outedge(cfg->exit(), 0);
			continue;
		}

		const llvm::TerminatorInst * instr = it->getTerminator();
		for (size_t n = 0; n < instr->getNumSuccessors(); n++)
			bbmap[&(*it)]->add_outedge(bbmap[instr->getSuccessor(n)], n);
	}

	return entry_block;
}

static void
convert_basic_block(const llvm::BasicBlock & basic_block, context & ctx)
{
	for (auto it = basic_block.begin(); it != basic_block.end(); it++)
		convert_instruction(*it, ctx.lookup_basic_block(&basic_block), ctx);
}

static void
convert_function(const llvm::Function & function, jlm::frontend::clg_node * clg_node)
{
	if (function.isDeclaration())
		return;

	std::vector<std::string> names;
	llvm::Function::ArgumentListType::const_iterator jt = function.getArgumentList().begin();
	for (; jt != function.getArgumentList().end(); jt++)
		names.push_back(jt->getName().str());
	names.push_back("_s_");

	std::vector<const jlm::frontend::variable*> arguments = clg_node->cfg_begin(names);
	const jlm::frontend::variable * state = arguments.back();
	jlm::frontend::cfg * cfg = clg_node->cfg();

	basic_block_map bbmap;
	frontend::basic_block * entry_block;
	entry_block = static_cast<frontend::basic_block*>(create_cfg_structure(function, cfg, bbmap));

	const jlm::frontend::variable * result = nullptr;
	if (function.getReturnType()->getTypeID() != llvm::Type::VoidTyID) {
		result = cfg->create_variable(*convert_type(*function.getReturnType()), "_r_");
		const frontend::variable * udef = create_undef_value(*function.getReturnType(), entry_block);
		assignment_tac(entry_block, result, udef);
	}

	context ctx(bbmap, entry_block, state, result);
	jt = function.getArgumentList().begin();
	for (size_t n = 0; jt != function.getArgumentList().end(); jt++, n++)
		ctx.insert_value(&(*jt), arguments[n]);

	auto it = function.getBasicBlockList().begin();
	for (; it != function.getBasicBlockList().end(); it++)
		convert_basic_block(*it, ctx);

	std::vector<const jlm::frontend::variable*> results;
	if (function.getReturnType()->getTypeID() != llvm::Type::VoidTyID)
		results.push_back(result);
	results.push_back(state);

	clg_node->cfg_end(results);
}

void
convert_module(const llvm::Module & module, jlm::frontend::clg & clg)
{
	JLM_DEBUG_ASSERT(clg.nnodes() == 0);

	function_map f_map;

	llvm::Module::FunctionListType::const_iterator it = module.getFunctionList().begin();
	for (; it != module.getFunctionList().end(); it++) {
		const llvm::Function & f = *it;
		jive::fct::type fcttype(dynamic_cast<const jive::fct::type&>(
			*convert_type(*f.getFunctionType())));
		f_map[&f] = clg.add_function(f.getName().str().c_str(), fcttype);
	}

	it = module.getFunctionList().begin();
	for (; it != module.getFunctionList().end(); it++)
		convert_function(*it, f_map[&(*it)]);
}

}

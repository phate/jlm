/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/constant.hpp>
#include <jlm/instruction.hpp>
#include <jlm/jlm.hpp>
#include <jlm/type.hpp>
#include <jlm/frontend/basic_block.hpp>
#include <jlm/frontend/cfg.hpp>
#include <jlm/frontend/cfg_node.hpp>

#include <jive/arch/memorytype.h>
#include <jive/vsdg/basetype.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace jlm
{

typedef std::unordered_map<const llvm::Function*, jlm::frontend::clg_node*> function_map;

static void
convert_basic_block(const llvm::BasicBlock & basic_block, const basic_block_map & bbmap,
	value_map & vmap, const jlm::frontend::output * state, const jlm::frontend::output * result)
{
	llvm::BasicBlock::const_iterator it;
	for (it = basic_block.begin(); it != basic_block.end(); it++)
		convert_instruction(*it, bbmap.find(&basic_block)->second, bbmap, vmap, state, result);
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

	std::vector<const jlm::frontend::output*> arguments = clg_node->cfg_begin(names);
	const jlm::frontend::output * state = arguments.back();
	jlm::frontend::cfg * cfg = clg_node->cfg();

	value_map vmap;
	jt = function.getArgumentList().begin();
	for (size_t n = 0; jt != function.getArgumentList().end(); jt++, n++)
		vmap[&(*jt)] = arguments[n];

	basic_block_map bbmap;
	llvm::Function::BasicBlockListType::const_iterator it = function.getBasicBlockList().begin();
	for (; it != function.getBasicBlockList().end(); it++)
			bbmap[&(*it)] = cfg->create_basic_block();

	jlm::frontend::basic_block * entry_block = bbmap[&function.getEntryBlock()];
	cfg->exit()->divert_inedges(entry_block);

	const jlm::frontend::output * result = nullptr;
	if (function.getReturnType()->getTypeID() != llvm::Type::VoidTyID)
		result = create_undef_value(*function.getReturnType(), entry_block);

	it = function.getBasicBlockList().begin();
	for (; it != function.getBasicBlockList().end(); it++)
		convert_basic_block(*it, bbmap, vmap, state, result);

	std::vector<const jlm::frontend::output*> results;
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

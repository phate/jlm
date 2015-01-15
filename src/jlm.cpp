/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/instruction.hpp>
#include <jlm/jlm.hpp>
#include <jlm/type.hpp>

#include <jive/arch/memorytype.h>
#include <jive/frontend/basic_block.h>
#include <jive/frontend/cfg.h>
#include <jive/frontend/cfg_node.h>
#include <jive/vsdg/basetype.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace jlm
{

typedef std::unordered_map<const llvm::Function*, jive::frontend::clg_node*> function_map;

static void
convert_basic_block(const llvm::BasicBlock & basic_block, const basic_block_map & bbmap,
	value_map & vmap, const jive::frontend::output * state)
{
	llvm::BasicBlock::const_iterator it;
	for (it = basic_block.begin(); it != basic_block.end(); it++)
		convert_instruction(*it, bbmap.find(&basic_block)->second, bbmap, vmap, state);
}

static void
convert_function(const llvm::Function & function, jive::frontend::clg_node * clg_node)
{
	if (function.isDeclaration())
		return;

	std::vector<std::string> names;
	llvm::Function::ArgumentListType::const_iterator jt = function.getArgumentList().begin();
	for (; jt != function.getArgumentList().end(); jt++)
		names.push_back(jt->getName().str());
	names.push_back("_s_");

	std::vector<const jive::frontend::output*> arguments = clg_node->cfg_begin(names);
	jive::frontend::cfg * cfg = clg_node->cfg();

	value_map vmap;
	jt = function.getArgumentList().begin();
	for (size_t n = 0; jt != function.getArgumentList().end(); jt++, n++)
		vmap[&(*jt)] = arguments[n];

	basic_block_map bbmap;
	llvm::Function::BasicBlockListType::const_iterator it = function.getBasicBlockList().begin();
	for (; it != function.getBasicBlockList().end(); it++)
			bbmap[&(*it)] = cfg->create_basic_block();

	cfg->exit()->divert_inedges(bbmap[&function.getEntryBlock()]);

	it = function.getBasicBlockList().begin();
	for (; it != function.getBasicBlockList().end(); it++)
		convert_basic_block(*it, bbmap, vmap, arguments.back());

	//FIXME: add results
	clg_node->cfg_end(std::vector<const jive::frontend::output*>());
}

void
convert_module(const llvm::Module & module, jive::frontend::clg & clg)
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

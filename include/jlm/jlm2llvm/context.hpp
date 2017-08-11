/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLM2LLVM_CONTEXT_HPP
#define JLM_JLM2LLVM_CONTEXT_HPP

#include <jlm/common.hpp>

#include <unordered_map>

namespace llvm {

class BasicBlock;
class Module;
class Value;

}

namespace jlm {

class cfg_node;
class module;
class variable;

namespace jlm2llvm {

class context final {
	typedef std::unordered_map<const cfg_node*, llvm::BasicBlock*>::const_iterator const_iterator;

public:
	inline
	context(jlm::module & jm, llvm::Module & lm)
	: lm_(lm)
	, jm_(jm)
	{}

	context(const context&) = delete;

	context(context&&) = delete;

	context &
	operator=(const context&) = delete;

	context &
	operator=(context&&) = delete;

	/*
		FIXME: It should be a const reference, but we still have to create variables to translate
		       expressions.
	*/
	inline jlm::module &
	jlm_module() const noexcept
	{
		return jm_;
	}

	inline llvm::Module &
	llvm_module() const noexcept
	{
		return lm_;
	}

	inline const_iterator
	begin() const
	{
		return nodes_.begin();
	}

	inline const_iterator
	end() const
	{
		return nodes_.end();
	}

	inline void
	insert(const jlm::cfg_node * node, llvm::BasicBlock * bb)
	{
		nodes_[node] = bb;
	}

	inline void
	insert(const jlm::variable * variable, llvm::Value * value)
	{
		variables_[variable] = value;
	}

	inline llvm::BasicBlock *
	basic_block(const jlm::cfg_node * node) const noexcept
	{
		auto it = nodes_.find(node);
		JLM_DEBUG_ASSERT(it != nodes_.end());
		return it->second;
	}

	inline llvm::Value *
	value(const jlm::variable * variable) const noexcept
	{
		auto it = variables_.find(variable);
		JLM_DEBUG_ASSERT(it != variables_.end());
		return it->second;
	}

private:
	llvm::Module & lm_;
	jlm::module & jm_;
	std::unordered_map<const jlm::variable*, llvm::Value*> variables_;
	std::unordered_map<const jlm::cfg_node*, llvm::BasicBlock*> nodes_;
};

}}

#endif

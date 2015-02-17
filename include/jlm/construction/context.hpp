/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_CONSTRUCTION_CONTEXT_HPP
#define JLM_CONSTRUCTION_CONTEXT_HPP

#include <unordered_map>

namespace llvm {
	class BasicBlock;
	class Value;
}

namespace jlm {
namespace frontend {
	class basic_block;
	class variable;
}

typedef std::unordered_map<const llvm::BasicBlock*, jlm::frontend::basic_block*> basic_block_map;

typedef std::unordered_map<const llvm::Value*, const jlm::frontend::variable*> value_map;

class context final {
public:
	inline
	context(
		const basic_block_map & bbmap,
		frontend::basic_block * entry_block,
		const frontend::variable * state,
		const frontend::variable * result)
	: bbmap_(bbmap)
	, entry_block_(entry_block)
	, state_(state)
	, result_(result)
	{}

	inline frontend::basic_block *
	entry_block() const noexcept
	{
		return entry_block_;
	}

	inline const frontend::variable *
	result() const noexcept
	{
		return result_;
	}

	inline const frontend::variable *
	state() const noexcept
	{
		return state_;
	}

	inline bool
	has_basic_block(const llvm::BasicBlock * bb) const noexcept
	{
		return bbmap_.find(bb) != bbmap_.end();
	}

	inline frontend::basic_block *
	lookup_basic_block(const llvm::BasicBlock * bb) const noexcept
	{
		JLM_DEBUG_ASSERT(has_basic_block(bb));
		return bbmap_.find(bb)->second;
	}

	inline const bool
	has_value(const llvm::Value * value) const noexcept
	{
		return vmap_.find(value) != vmap_.end();
	}

	inline const frontend::variable *
	lookup_value(const llvm::Value * value) const noexcept
	{
		JLM_DEBUG_ASSERT(has_value(value));
		return vmap_.find(value)->second;
	}

	inline void
	insert_value(const llvm::Value * value, const frontend::variable * variable)
	{
		vmap_[value] = variable;
	}

private:
	const basic_block_map & bbmap_;
	frontend::basic_block * entry_block_;
	const frontend::variable * state_;
	const frontend::variable * result_;
	std::unordered_map<const llvm::Value *, const frontend::variable*> vmap_;
};

}

#endif

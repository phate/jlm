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

class basic_block_map final {
public:

	inline bool
	has_basic_block(const llvm::BasicBlock * bb) const noexcept
	{
		return llvm2jlm_.find(bb) != llvm2jlm_.end();
	}

	inline bool
	has_basic_block(const frontend::basic_block * bb) const noexcept
	{
		return jlm2llvm_.find(bb) != jlm2llvm_.end();
	}

	inline frontend::basic_block *
	lookup_basic_block(const llvm::BasicBlock * bb) const noexcept
	{
		JLM_DEBUG_ASSERT(has_basic_block(bb));
		return llvm2jlm_.find(bb)->second;
	}

	inline const llvm::BasicBlock *
	lookup_basic_block(const frontend::basic_block * bb) const noexcept
	{
		JLM_DEBUG_ASSERT(has_basic_block(bb));
		return jlm2llvm_.find(bb)->second;
	}

	inline void
	insert_basic_block(const llvm::BasicBlock * bb1, frontend::basic_block * bb2)
	{
		JLM_DEBUG_ASSERT(!has_basic_block(bb1));
		JLM_DEBUG_ASSERT(!has_basic_block(bb2));
		llvm2jlm_[bb1] = bb2;
		jlm2llvm_[bb2] = bb1;
	}

	frontend::basic_block *
	operator[](const llvm::BasicBlock * bb) const
	{
		return lookup_basic_block(bb);
	}

	const llvm::BasicBlock *
	operator[](const frontend::basic_block * bb) const
	{
		return lookup_basic_block(bb);
	}

private:
	std::unordered_map<const llvm::BasicBlock*, frontend::basic_block*> llvm2jlm_;
	std::unordered_map<const frontend::basic_block*, const llvm::BasicBlock*> jlm2llvm_;
};

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
		return bbmap_.has_basic_block(bb);
	}

	inline bool
	has_basic_block(const frontend::basic_block * bb) const noexcept
	{
		return bbmap_.has_basic_block(bb);
	}

	inline frontend::basic_block *
	lookup_basic_block(const llvm::BasicBlock * bb) const noexcept
	{
		return bbmap_.lookup_basic_block(bb);
	}

	inline const llvm::BasicBlock *
	lookup_basic_block(const frontend::basic_block * bb) const noexcept
	{
		return bbmap_.lookup_basic_block(bb);
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

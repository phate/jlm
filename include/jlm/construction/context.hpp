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

class basic_block;
class variable;

class basic_block_map final {
public:

	inline bool
	has_basic_block(const llvm::BasicBlock * bb) const noexcept
	{
		return llvm2jlm_.find(bb) != llvm2jlm_.end();
	}

	inline bool
	has_basic_block(const basic_block * bb) const noexcept
	{
		return jlm2llvm_.find(bb) != jlm2llvm_.end();
	}

	inline basic_block *
	lookup_basic_block(const llvm::BasicBlock * bb) const noexcept
	{
		JLM_DEBUG_ASSERT(has_basic_block(bb));
		return llvm2jlm_.find(bb)->second;
	}

	inline const llvm::BasicBlock *
	lookup_basic_block(const basic_block * bb) const noexcept
	{
		JLM_DEBUG_ASSERT(has_basic_block(bb));
		return jlm2llvm_.find(bb)->second;
	}

	inline void
	insert_basic_block(const llvm::BasicBlock * bb1, basic_block * bb2)
	{
		JLM_DEBUG_ASSERT(!has_basic_block(bb1));
		JLM_DEBUG_ASSERT(!has_basic_block(bb2));
		llvm2jlm_[bb1] = bb2;
		jlm2llvm_[bb2] = bb1;
	}

	basic_block *
	operator[](const llvm::BasicBlock * bb) const
	{
		return lookup_basic_block(bb);
	}

	const llvm::BasicBlock *
	operator[](const basic_block * bb) const
	{
		return lookup_basic_block(bb);
	}

private:
	std::unordered_map<const llvm::BasicBlock*, basic_block*> llvm2jlm_;
	std::unordered_map<const basic_block*, const llvm::BasicBlock*> jlm2llvm_;
};

typedef std::unordered_map<const llvm::Value*, const jlm::variable*> value_map;

class context final {
public:
	inline
	context(
		const basic_block_map & bbmap,
		basic_block * entry_block,
		const variable * state,
		const variable * result)
	: bbmap_(bbmap)
	, entry_block_(entry_block)
	, state_(state)
	, result_(result)
	{}

	inline basic_block *
	entry_block() const noexcept
	{
		return entry_block_;
	}

	inline const variable *
	result() const noexcept
	{
		return result_;
	}

	inline const variable *
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
	has_basic_block(const basic_block * bb) const noexcept
	{
		return bbmap_.has_basic_block(bb);
	}

	inline basic_block *
	lookup_basic_block(const llvm::BasicBlock * bb) const noexcept
	{
		return bbmap_.lookup_basic_block(bb);
	}

	inline const llvm::BasicBlock *
	lookup_basic_block(const basic_block * bb) const noexcept
	{
		return bbmap_.lookup_basic_block(bb);
	}

	inline const bool
	has_value(const llvm::Value * value) const noexcept
	{
		return vmap_.find(value) != vmap_.end();
	}

	inline const variable *
	lookup_value(const llvm::Value * value) const noexcept
	{
		JLM_DEBUG_ASSERT(has_value(value));
		return vmap_.find(value)->second;
	}

	inline void
	insert_value(const llvm::Value * value, const variable * variable)
	{
		JLM_DEBUG_ASSERT(!has_value(value));
		vmap_[value] = variable;
	}

private:
	const basic_block_map & bbmap_;
	basic_block * entry_block_;
	const variable * state_;
	const variable * result_;
	std::unordered_map<const llvm::Value *, const variable*> vmap_;
};

}

#endif

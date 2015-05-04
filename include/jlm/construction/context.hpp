/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_CONSTRUCTION_CONTEXT_HPP
#define JLM_CONSTRUCTION_CONTEXT_HPP

#include <jlm/construction/type.hpp>

#include <jive/types/record/rcdtype.h>

#include <llvm/IR/DerivedTypes.h>

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

class context final {
public:
	inline
	context()
		: entry_block_(nullptr)
		, state_(nullptr)
		, result_(nullptr)
	{}

	inline basic_block *
	entry_block() const noexcept
	{
		return entry_block_;
	}

	inline void
	set_entry_block(basic_block * entry_block)
	{
		entry_block_ = entry_block;
	}

	inline const variable *
	result() const noexcept
	{
		return result_;
	}

	inline void
	set_result(const variable * result)
	{
		result_ = result;
	}

	inline const variable *
	state() const noexcept
	{
		return state_;
	}

	inline void
	set_state(const variable * state)
	{
		state_ = state;
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

	inline void
	set_basic_block_map(const basic_block_map & bbmap)
	{
		bbmap_ = bbmap;
	}

	inline const bool
	has_value(const llvm::Value * value) const noexcept
	{
		return vmap_.find(value) != vmap_.end();
	}

	inline variable *
	lookup_value(const llvm::Value * value) const noexcept
	{
		JLM_DEBUG_ASSERT(has_value(value));
		return vmap_.find(value)->second;
	}

	inline void
	insert_value(const llvm::Value * value, variable * variable)
	{
		JLM_DEBUG_ASSERT(!has_value(value));
		vmap_[value] = variable;
	}

	inline std::shared_ptr<const jive::rcd::declaration> &
	lookup_declaration(const llvm::StructType * type)
	{
		auto it = declarations_.find(type);
		if (it != declarations_.end())
			return it->second;

		std::vector<const jive::value::type*> types_;
		std::vector<std::unique_ptr<jive::base::type>> types;
		for (size_t n = 0; n < type->getNumElements(); n++) {
			types.emplace_back(convert_type(type->getElementType(n), *this));
			types_.push_back(dynamic_cast<const jive::value::type*>((types.back().get())));
		}

		std::shared_ptr<const jive::rcd::declaration> declaration(new jive::rcd::declaration(types_));
		declarations_[type] = declaration;

		return declarations_[type];
	}

private:
	basic_block_map bbmap_;
	basic_block * entry_block_;
	const variable * state_;
	const variable * result_;
	std::unordered_map<const llvm::Value *, variable*> vmap_;
	std::unordered_map<
		const llvm::StructType*,
		std::shared_ptr<const jive::rcd::declaration>> declarations_;
};

}

#endif

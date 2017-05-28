/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_CONSTRUCTION_CONTEXT_HPP
#define JLM_CONSTRUCTION_CONTEXT_HPP

#include <jlm/construction/type.hpp>
#include <jlm/IR/cfg_node.hpp>

#include <jive/types/record/rcdtype.h>
#include <llvm/IR/DerivedTypes.h>

#include <unordered_map>

namespace llvm {
	class BasicBlock;
	class Function;
	class Value;
}

namespace jlm {

class cfg;
class cfg_node;
class clg_node;
class variable;

class basic_block_map final {
public:
	inline bool
	has_basic_block(const llvm::BasicBlock * bb) const noexcept
	{
		return llvm2jlm_.find(bb) != llvm2jlm_.end();
	}

	inline bool
	has_basic_block(const cfg_node * bb) const noexcept
	{
		return jlm2llvm_.find(bb) != jlm2llvm_.end();
	}

	inline cfg_node *
	lookup_basic_block(const llvm::BasicBlock * bb) const noexcept
	{
		JLM_DEBUG_ASSERT(has_basic_block(bb));
		return llvm2jlm_.find(bb)->second;
	}

	inline const llvm::BasicBlock *
	lookup_basic_block(const cfg_node * bb) const noexcept
	{
		JLM_DEBUG_ASSERT(has_basic_block(bb));
		return jlm2llvm_.find(bb)->second;
	}

	inline void
	insert_basic_block(const llvm::BasicBlock * bb1, cfg_node * bb2)
	{
		JLM_DEBUG_ASSERT(!has_basic_block(bb1));
		JLM_DEBUG_ASSERT(!has_basic_block(bb2));
		llvm2jlm_[bb1] = bb2;
		jlm2llvm_[bb2] = bb1;
	}

	cfg_node *
	operator[](const llvm::BasicBlock * bb) const
	{
		return lookup_basic_block(bb);
	}

	const llvm::BasicBlock *
	operator[](const cfg_node * bb) const
	{
		return lookup_basic_block(bb);
	}

private:
	std::unordered_map<const llvm::BasicBlock*, cfg_node*> llvm2jlm_;
	std::unordered_map<const cfg_node*, const llvm::BasicBlock*> jlm2llvm_;
};

class context final {
public:
	inline
	context()
		: entry_block_(nullptr)
	{}

	inline cfg_node *
	entry_block() const noexcept
	{
		return entry_block_;
	}

	inline void
	set_entry_block(cfg_node * entry_block)
	{
		entry_block_ = entry_block;
	}

	inline std::shared_ptr<const variable>
	result() const noexcept
	{
		return result_;
	}

	inline void
	set_result(const std::shared_ptr<const variable> & result)
	{
		result_ = result;
	}

	inline std::shared_ptr<const variable>
	state() const noexcept
	{
		return state_;
	}

	inline void
	set_state(const std::shared_ptr<const variable> & state)
	{
		state_ = state;
	}

	inline bool
	has_basic_block(const llvm::BasicBlock * bb) const noexcept
	{
		return bbmap_.has_basic_block(bb);
	}

	inline bool
	has_basic_block(const cfg_node * bb) const noexcept
	{
		return bbmap_.has_basic_block(bb);
	}

	inline cfg_node *
	lookup_basic_block(const llvm::BasicBlock * bb) const noexcept
	{
		return bbmap_.lookup_basic_block(bb);
	}

	inline const llvm::BasicBlock *
	lookup_basic_block(const cfg_node * bb) const noexcept
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

	inline std::shared_ptr<variable>
	lookup_value(const llvm::Value * value) const noexcept
	{
		JLM_DEBUG_ASSERT(has_value(value));
		return vmap_.find(value)->second;
	}

	inline void
	insert_value(const llvm::Value * value, const std::shared_ptr<variable> & variable)
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

	inline jlm::clg_node *
	lookup_function(const llvm::Function * f) const noexcept
	{
		auto it = fmap_.find(f);
		return it != fmap_.end() ? it->second : nullptr;
	}

	inline void
	insert_function(const llvm::Function * f, clg_node * n)
	{
		fmap_[f] = n;
	}

	inline jlm::cfg *
	cfg() const noexcept
	{
		return entry_block_->cfg();
	}

private:
	basic_block_map bbmap_;
	cfg_node * entry_block_;
	std::shared_ptr<const variable> state_;
	std::shared_ptr<const variable> result_;
	std::unordered_map<const llvm::Function*, jlm::clg_node*> fmap_;
	std::unordered_map<const llvm::Value *, std::shared_ptr<variable>> vmap_;
	std::unordered_map<
		const llvm::StructType*,
		std::shared_ptr<const jive::rcd::declaration>> declarations_;
};

}

#endif

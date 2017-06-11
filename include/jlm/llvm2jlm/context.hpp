/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM2JLM_CONTEXT_HPP
#define JLM_LLVM2JLM_CONTEXT_HPP

#include <jlm/IR/cfg_node.hpp>
#include <jlm/IR/expression.hpp>
#include <jlm/IR/module.hpp>
#include <jlm/IR/tac.hpp>
#include <jlm/llvm2jlm/type.hpp>

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
class module;
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
	context(jlm::module & module)
		: module_(module)
		, entry_block_(nullptr)
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

	inline jlm::cfg *
	cfg() const noexcept
	{
		return entry_block_->cfg();
	}

	inline jlm::module &
	module() const noexcept
	{
		return module_;
	}

private:
	jlm::module & module_;
	basic_block_map bbmap_;
	cfg_node * entry_block_;
	const variable * state_;
	const variable * result_;
	std::unordered_map<const llvm::Value *, const variable *> vmap_;
	std::unordered_map<
		const llvm::StructType*,
		std::shared_ptr<const jive::rcd::declaration>> declarations_;
};

}

static inline std::vector<std::unique_ptr<jlm::tac>>
expr2tacs(const jlm::expr & e, const jlm::context & ctx)
{
	std::function<const jlm::variable *(
		const jlm::expr &,
		const jlm::variable *,
		std::vector<std::unique_ptr<jlm::tac>>&)
	> append = [&](
		const jlm::expr & e,
		const jlm::variable * result,
		std::vector<std::unique_ptr<jlm::tac>> & tacs)
	{
		std::vector<const jlm::variable *> operands;
		for (size_t n = 0; n < e.noperands(); n++) {
			auto v = ctx.module().create_variable(e.operand(n).type(), false);
			operands.push_back(append(e.operand(n), v, tacs));
		}

		tacs.emplace_back(create_tac(e.operation(), operands, {result}));
		return tacs.back()->output(0);
	};

	std::vector<std::unique_ptr<jlm::tac>> tacs;
	append(e, ctx.module().create_variable(e.type(), false), tacs);
	return tacs;
}

#endif

/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_MODULE_HPP
#define JLM_IR_MODULE_HPP

#include <jlm/ir/clg.hpp>
#include <jlm/ir/expression.hpp>
#include <jlm/ir/tac.hpp>

namespace jlm {

class module final {
	typedef std::unordered_map<
		const jlm::variable*,
		std::unique_ptr<const expr>
	>::const_iterator const_iterator;
public:
	inline
	~module()
	{}

	inline
	module(
		const std::string & target_triple,
		const std::string & data_layout) noexcept
	: data_layout_(data_layout)
	, target_triple_(target_triple)
	{}

	inline jlm::clg &
	clg() noexcept
	{
		return clg_;
	}

	inline const jlm::clg &
	clg() const noexcept
	{
		return clg_;
	}

	inline void
	add_global_variable(const jlm::variable * v, std::unique_ptr<const expr> e)
	{
		globals_[v] = std::move(e);
	}

	const_iterator
	begin() const
	{
		return globals_.begin();
	}

	const_iterator
	end() const
	{
		return globals_.end();
	}

	/*
		FIXME: tacs are supposed to be the owners of tacvariable. This is going to be removed
		       again.
	*/
	inline jlm::tacvariable *
	create_tacvariable(const jive::base::type & type)
	{
		static uint64_t c = 0;
		auto v = jlm::create_tacvariable(type, strfmt("tv", c++));
		auto pv = v.get();
		variables_.insert(std::move(v));
		return static_cast<tacvariable*>(pv);
	}

	inline jlm::variable *
	create_variable(const jive::base::type & type, const std::string & name, bool exported)
	{
		auto v = std::make_unique<jlm::variable>(type, name, exported);
		auto pv = v.get();
		variables_.insert(std::move(v));
		return pv;
	}

	inline jlm::variable *
	create_variable(const jive::base::type & type, bool exported)
	{
		static uint64_t c = 0;
		auto v = std::make_unique<jlm::variable>(type, strfmt("v", c++), exported);
		auto pv = v.get();
		variables_.insert(std::move(v));
		return pv;
	}

	inline jlm::variable *
	create_variable(clg_node * node)
	{
		JLM_DEBUG_ASSERT(!variable(node));

		auto v = std::unique_ptr<jlm::variable>(new fctvariable(node));
		auto pv = v.get();
		functions_[node] = pv;
		variables_.insert(std::move(v));
		return pv;
	}

	const jlm::variable *
	variable(const clg_node * node) const noexcept
	{
		auto it = functions_.find(node);
		return it != functions_.end() ? it->second : nullptr;
	}

	inline const std::string &
	target_triple() const noexcept
	{
		return target_triple_;
	}

	inline const std::string &
	data_layout() const noexcept
	{
		return data_layout_;
	}

private:
	jlm::clg clg_;
	std::string data_layout_;
	std::string target_triple_;
	std::unordered_set<std::unique_ptr<jlm::variable>> variables_;
	std::unordered_map<const clg_node*, const jlm::variable*> functions_;
	std::unordered_map<const jlm::variable*, std::unique_ptr<const jlm::expr>> globals_;
};

}

#endif

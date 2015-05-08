/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_CLG_H
#define JLM_IR_CLG_H

#include <jlm/IR/cfg.hpp>
#include <jlm/IR/variable.hpp>

#include <jive/types/function/fcttype.h>

#include <unordered_map>
#include <unordered_set>

namespace jlm {

class clg_node;

/* clg */

class clg final {
public:
	inline
	~clg()
	{}

	inline
	clg() noexcept
	{}

	clg_node *
	add_function(const char * name, jive::fct::type & type, bool exported);

	clg_node *
	lookup_function(const std::string & name) const;

	inline clg_node *
	lookup_function(const char * name) const
	{
		return lookup_function(std::string(name));
	}

	std::vector<clg_node*>
	nodes() const;

	inline size_t
	nnodes() const noexcept
	{
		return nodes_.size();
	}

	std::vector<std::unordered_set<const clg_node*>>
	find_sccs() const;

	std::string
	to_string() const;

private:
	std::unordered_map<std::string, std::unique_ptr<clg_node>> nodes_;
};

/* clg node */

class output;

class clg_node final : public global_variable {
public:
	virtual
	~clg_node() noexcept;

private:
	inline
	clg_node(jlm::clg & clg, const char * name, jive::fct::type & type, bool exported)
		: global_variable(type, name, exported)
		, cfg_(nullptr)
		, clg_(clg)
	{}

public:
	inline jlm::cfg *
	cfg() const noexcept
	{
		return cfg_.get();
	}

	inline jlm::clg &
	clg() const noexcept
	{
		return clg_;
	}

	virtual const jive::fct::type &
	type() const noexcept override;

	void
	add_call(const clg_node * callee)
	{
		calls_.insert(callee);
	}

	const std::unordered_set<const clg_node*> &
	calls() const
	{
		return calls_;
	}

	bool
	is_selfrecursive() const noexcept
	{
		if (calls_.find(this) != calls_.end())
			return true;

		return false;
	}

	std::vector<variable*>
	cfg_begin(const std::vector<std::string> & names);

	void
	cfg_end(const std::vector<variable*> & results);

private:
	std::unique_ptr<jlm::cfg> cfg_;
	jlm::clg & clg_;
	std::unordered_set<const clg_node*> calls_;

	friend jlm::clg_node * jlm::clg::add_function(const char * name,
		jive::fct::type & type, bool exported);
};

}

#endif

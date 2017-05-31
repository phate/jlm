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

class clg_node final {
public:
	inline
	~clg_node() noexcept
	{}

private:
	inline
	clg_node(jlm::clg & clg, const char * name, jive::fct::type & type, bool exported)
	: exported_(exported)
	, clg_(clg)
	, name_(name)
	, cfg_(nullptr)
	, type_(std::move(type.copy()))
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

	const jive::fct::type &
	type() const noexcept
	{
		return *static_cast<const jive::fct::type*>(type_.get());
	}

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

	inline bool
	exported() const noexcept
	{
		return exported_;
	}

	inline const std::string &
	name() const noexcept
	{
		return name_;
	}

	inline void
	add_cfg(std::unique_ptr<jlm::cfg> cfg)
	{
		cfg_ = std::move(cfg);
	}

private:
	bool exported_;
	jlm::clg & clg_;
	std::string name_;
	std::unique_ptr<jlm::cfg> cfg_;
	std::unique_ptr<jive::base::type> type_;
	std::unordered_set<const clg_node*> calls_;

	friend jlm::clg_node * jlm::clg::add_function(const char * name,
		jive::fct::type & type, bool exported);
};

class function_variable final : public variable {
public:
	virtual
	~function_variable();

	inline
	function_variable(clg_node * node)
	: variable(node->type(), node->name(), node->exported())
	, node_(node)
	{}

	inline jlm::clg_node *
	function() const noexcept
	{
		return node_;
	}

private:
	jlm::clg_node * node_;
};

}

#endif

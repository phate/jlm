/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_CLG_H
#define JLM_IR_CLG_H

#include <jlm/IR/cfg.hpp>

#include <jive/types/function/fcttype.h>

#include <unordered_map>
#include <unordered_set>

namespace jive {
	class buffer;
}

namespace jlm {
namespace frontend {

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
	add_function(const char * name, jive::fct::type & type);

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

private:
	std::unordered_map<std::string, std::unique_ptr<clg_node>> nodes_;
};

/* clg node */

class output;

class clg_node final {
public:
	inline
	~clg_node()
	{}

private:
	inline
	clg_node(jlm::frontend::clg & clg, const char * name, jive::fct::type & type)
		: name_(name)
		, cfg_(nullptr)
		, clg_(clg)
		, type_(type.copy())
	{}

public:
	inline jlm::frontend::cfg *
	cfg() const noexcept
	{
		return cfg_.get();
	}

	inline jlm::frontend::clg &
	clg() const noexcept
	{
		return clg_;
	}

	inline const std::string &
	name() const noexcept
	{
		return name_;
	}

	inline const jive::fct::type &
	type() const noexcept
	{
		return *type_;
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

	std::vector<const output*>
	cfg_begin(const std::vector<std::string> & names);

	void
	cfg_end(const std::vector<const variable*> & results);

private:
	std::string name_;
	std::unique_ptr<jlm::frontend::cfg> cfg_;
	jlm::frontend::clg & clg_;
	std::unique_ptr<jive::fct::type> type_;
	std::unordered_set<const clg_node*> calls_;

	friend jlm::frontend::clg_node * jlm::frontend::clg::add_function(const char * name,
		jive::fct::type & type);
};

}
}

void
jive_clg_convert_dot(const jlm::frontend::clg & self, jive::buffer & buffer);

void
jive_clg_view(const jlm::frontend::clg & self);

#endif

/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_CALLGRAPH_H
#define JLM_IR_CALLGRAPH_H

#include <jlm/ir/cfg.hpp>
#include <jlm/ir/types.hpp>
#include <jlm/ir/variable.hpp>

#include <jive/types/function/fcttype.h>

#include <unordered_map>
#include <unordered_set>

namespace jlm {

class callgraph_node;

/* callgraph */

class callgraph final {
	class const_iterator {
	public:
		inline
		const_iterator(const std::unordered_map<
			std::string,
			std::unique_ptr<callgraph_node>>::const_iterator & it)
		: it_(it)
		{}

		inline bool
		operator==(const const_iterator & other) const noexcept
		{
			return it_ == other.it_;
		}

		inline bool
		operator!=(const const_iterator & other) const noexcept
		{
			return !(*this == other);
		}

		inline const const_iterator &
		operator++() noexcept
		{
			++it_;
			return *this;
		}

		inline const const_iterator
		operator++(int) noexcept
		{
			const_iterator tmp(it_);
			it_++;
			return tmp;
		}

		inline const callgraph_node *
		node() const noexcept
		{
			return it_->second.get();
		}

		inline const callgraph_node &
		operator*() const noexcept
		{
			return *node();
		}

		inline const callgraph_node *
		operator->() const noexcept
		{
			return node();
		}

	private:
		std::unordered_map<
			std::string,
			std::unique_ptr<callgraph_node>
		>::const_iterator it_;
	};

public:
	inline
	~callgraph()
	{}

	inline
	callgraph() noexcept
	{}

	inline const_iterator
	begin() const noexcept
	{
		return const_iterator(nodes_.begin());
	}

	inline const_iterator
	end() const noexcept
	{
		return const_iterator(nodes_.end());
	}

	void
	add_function(std::unique_ptr<callgraph_node> node);

	callgraph_node *
	lookup_function(const std::string & name) const;

	inline callgraph_node *
	lookup_function(const char * name) const
	{
		return lookup_function(std::string(name));
	}

	std::vector<callgraph_node*>
	nodes() const;

	inline size_t
	nnodes() const noexcept
	{
		return nodes_.size();
	}

	std::vector<std::unordered_set<const callgraph_node*>>
	find_sccs() const;

private:
	std::unordered_map<
		std::string,
		std::unique_ptr<callgraph_node>
	> nodes_;
};

/* clg node */

class output;

class callgraph_node {
	typedef std::unordered_set<const callgraph_node*>::const_iterator const_iterator;
public:
	virtual
	~callgraph_node() noexcept;

protected:
	inline
	callgraph_node(jlm::callgraph & clg)
	: clg_(clg)
	{}

public:
	inline jlm::callgraph &
	clg() const noexcept
	{
		return clg_;
	}

	void
	add_dependency(const callgraph_node * dep)
	{
		dependencies_.insert(dep);
	}

	inline const_iterator
	begin() const
	{
		return dependencies_.begin();
	}

	inline const_iterator
	end() const
	{
		return dependencies_.end();
	}

	bool
	is_selfrecursive() const noexcept
	{
		if (dependencies_.find(this) != dependencies_.end())
			return true;

		return false;
	}

	virtual const std::string &
	name() const noexcept = 0;

	virtual const jive::type &
	type() const noexcept = 0;

private:
	jlm::callgraph & clg_;
	std::unordered_set<const callgraph_node*> dependencies_;
};

class function_node final : public callgraph_node {
public:
	virtual
	~function_node() noexcept;

private:
	inline
	function_node(
		jlm::callgraph & clg,
		const std::string & name,
		const jive::fct::type & type,
		bool exported)
	: callgraph_node(clg)
	, type_(type)
	, exported_(exported)
	, name_(name)
	{}

public:
	inline jlm::cfg *
	cfg() const noexcept
	{
		return cfg_.get();
	}

	virtual const jive::type &
	type() const noexcept override;

	const jive::fct::type &
	fcttype() const noexcept
	{
		return *static_cast<const jive::fct::type*>(&type_.pointee_type());
	}

	inline bool
	exported() const noexcept
	{
		return exported_;
	}

	const std::string &
	name() const noexcept override;

	inline void
	add_cfg(std::unique_ptr<jlm::cfg> cfg)
	{
		cfg_ = std::move(cfg);
	}

	static inline function_node *
	create(
		jlm::callgraph & clg,
		const std::string & name,
		const jive::fct::type & type,
		bool exported)
	{
		std::unique_ptr<function_node> node(new function_node(clg, name, type, exported));
		auto tmp = node.get();
		clg.add_function(std::move(node));
		return tmp;
	}

private:
	ptrtype type_;
	bool exported_;
	std::string name_;
	std::unique_ptr<jlm::cfg> cfg_;
};

class fctvariable final : public gblvariable {
public:
	virtual
	~fctvariable();

	inline
	fctvariable(function_node * node, const jlm::linkage & linkage)
	: gblvariable(node->type(), node->name(), linkage)
	, node_(node)
	{}

	inline function_node *
	function() const noexcept
	{
		return node_;
	}

private:
	function_node * node_;
};

static inline bool
is_fctvariable(const jlm::variable * v)
{
	return dynamic_cast<const jlm::fctvariable*>(v) != nullptr;
}

}

#endif

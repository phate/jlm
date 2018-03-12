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

class callgraph_node final {
public:
	inline
	~callgraph_node() noexcept
	{}

private:
	inline
	callgraph_node(
		jlm::callgraph & clg,
		const std::string & name,
		const jive::fct::type & type,
		bool exported)
	: type_(type)
	, exported_(exported)
	, name_(name)
	, clg_(clg)
	, cfg_(nullptr)
	{}

public:
	inline jlm::cfg *
	cfg() const noexcept
	{
		return cfg_.get();
	}

	inline jlm::callgraph &
	clg() const noexcept
	{
		return clg_;
	}

	const ptrtype &
	type() const noexcept
	{
		return type_;
	}

	const jive::fct::type &
	fcttype() const noexcept
	{
		return *static_cast<const jive::fct::type*>(&type().pointee_type());
	}

	void
	add_call(const callgraph_node * callee)
	{
		calls_.insert(callee);
	}

	const std::unordered_set<const callgraph_node*> &
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

	static inline callgraph_node *
	create(
		jlm::callgraph & clg,
		const std::string & name,
		const jive::fct::type & type,
		bool exported)
	{
		std::unique_ptr<callgraph_node> node(new callgraph_node(clg, name, type, exported));
		auto tmp = node.get();
		clg.add_function(std::move(node));
		return tmp;
	}

private:
	ptrtype type_;
	bool exported_;
	std::string name_;
	jlm::callgraph & clg_;
	std::unique_ptr<jlm::cfg> cfg_;
	std::unordered_set<const callgraph_node*> calls_;
};

class fctvariable final : public gblvariable {
public:
	virtual
	~fctvariable();

	inline
	fctvariable(callgraph_node * node, const jlm::linkage & linkage)
	: gblvariable(node->type(), node->name(), linkage)
	, node_(node)
	{}

	inline callgraph_node *
	function() const noexcept
	{
		return node_;
	}

private:
	callgraph_node * node_;
};

static inline bool
is_fctvariable(const jlm::variable * v)
{
	return dynamic_cast<const jlm::fctvariable*>(v) != nullptr;
}

}

#endif

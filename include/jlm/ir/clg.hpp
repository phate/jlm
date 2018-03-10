/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_CLG_H
#define JLM_IR_CLG_H

#include <jlm/ir/cfg.hpp>
#include <jlm/ir/types.hpp>
#include <jlm/ir/variable.hpp>

#include <jive/types/function/fcttype.h>

#include <unordered_map>
#include <unordered_set>

namespace jlm {

class clg_node;

/* clg */

class clg final {
	class const_iterator {
	public:
		inline
		const_iterator(const std::unordered_map<
			std::string,
			std::unique_ptr<clg_node>>::const_iterator & it)
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

		inline const clg_node *
		node() const noexcept
		{
			return it_->second.get();
		}

		inline const clg_node &
		operator*() const noexcept
		{
			return *node();
		}

		inline const clg_node *
		operator->() const noexcept
		{
			return node();
		}

	private:
		std::unordered_map<std::string, std::unique_ptr<clg_node>>::const_iterator it_;
	};

public:
	inline
	~clg()
	{}

	inline
	clg() noexcept
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
	add_function(std::unique_ptr<jlm::clg_node> node);

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
	clg_node(jlm::clg & clg, const std::string & name, const jive::fct::type & type, bool exported)
	: type_(type)
	, exported_(exported)
	, clg_(clg)
	, name_(name)
	, cfg_(nullptr)
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

	static inline clg_node *
	create(jlm::clg & clg, const std::string & name, const jive::fct::type & type, bool exported)
	{
		std::unique_ptr<jlm::clg_node> node(new clg_node(clg, name, type, exported));
		auto tmp = node.get();
		clg.add_function(std::move(node));
		return tmp;
	}

private:
	ptrtype type_;
	bool exported_;
	jlm::clg & clg_;
	std::string name_;
	std::unique_ptr<jlm::cfg> cfg_;
	std::unordered_set<const clg_node*> calls_;
};

class fctvariable final : public gblvariable {
public:
	virtual
	~fctvariable();

	inline
	fctvariable(clg_node * node, const jlm::linkage & linkage)
	: gblvariable(node->type(), node->name(), linkage)
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

static inline bool
is_fctvariable(const jlm::variable * v)
{
	return dynamic_cast<const jlm::fctvariable*>(v) != nullptr;
}

}

#endif

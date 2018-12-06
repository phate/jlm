/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_IPGRAPH_H
#define JLM_IR_IPGRAPH_H

#include <jlm/jlm/ir/cfg.hpp>
#include <jlm/jlm/ir/tac.hpp>
#include <jlm/jlm/ir/types.hpp>
#include <jlm/jlm/ir/variable.hpp>

#include <jive/types/function.h>

#include <unordered_map>
#include <unordered_set>

namespace jlm {

class ipgraph_node;

/* inter-procedure graph */

class ipgraph final {
	class const_iterator {
	public:
		inline
		const_iterator(
			const std::vector<std::unique_ptr<ipgraph_node>>::const_iterator & it)
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

		inline const ipgraph_node *
		node() const noexcept
		{
			return it_->get();
		}

		inline const ipgraph_node &
		operator*() const noexcept
		{
			return *node();
		}

		inline const ipgraph_node *
		operator->() const noexcept
		{
			return node();
		}

	private:
		std::vector<std::unique_ptr<ipgraph_node>>::const_iterator it_;
	};

public:
	inline
	~ipgraph()
	{}

	inline
	ipgraph() noexcept
	{}

	ipgraph(const ipgraph &) = delete;

	ipgraph(ipgraph &&) = delete;

	ipgraph &
	operator=(const ipgraph &) = delete;

	ipgraph &
	operator=(ipgraph &&) = delete;

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
	add_node(std::unique_ptr<ipgraph_node> node);

	inline size_t
	nnodes() const noexcept
	{
		return nodes_.size();
	}

	std::vector<std::unordered_set<const ipgraph_node*>>
	find_sccs() const;

private:
	std::vector<std::unique_ptr<ipgraph_node>> nodes_;
};

/* clg node */

class output;

class ipgraph_node {
	typedef std::unordered_set<const ipgraph_node*>::const_iterator const_iterator;
public:
	virtual
	~ipgraph_node() noexcept;

protected:
	inline
	ipgraph_node(jlm::ipgraph & clg)
	: clg_(clg)
	{}

public:
	inline jlm::ipgraph &
	clg() const noexcept
	{
		return clg_;
	}

	void
	add_dependency(const ipgraph_node * dep)
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

	virtual const jlm::linkage &
	linkage() const noexcept = 0;

private:
	jlm::ipgraph & clg_;
	std::unordered_set<const ipgraph_node*> dependencies_;
};

class function_node final : public ipgraph_node {
public:
	virtual
	~function_node() noexcept;

private:
	inline
	function_node(
		jlm::ipgraph & clg,
		const std::string & name,
		const jive::fcttype & type,
		const jlm::linkage & linkage)
	: ipgraph_node(clg)
	, type_(type)
	, name_(name)
	, linkage_(linkage)
	{}

public:
	inline jlm::cfg *
	cfg() const noexcept
	{
		return cfg_.get();
	}

	virtual const jive::type &
	type() const noexcept override;

	const jive::fcttype &
	fcttype() const noexcept
	{
		return *static_cast<const jive::fcttype*>(&type_.pointee_type());
	}

	virtual const jlm::linkage &
	linkage() const noexcept override;

	const std::string &
	name() const noexcept override;

	inline void
	add_cfg(std::unique_ptr<jlm::cfg> cfg)
	{
		cfg_ = std::move(cfg);
	}

	static inline function_node *
	create(
		jlm::ipgraph & clg,
		const std::string & name,
		const jive::fcttype & type,
		const jlm::linkage & linkage)
	{
		std::unique_ptr<function_node> node(new function_node(clg, name, type, linkage));
		auto tmp = node.get();
		clg.add_node(std::move(node));
		return tmp;
	}

private:
	ptrtype type_;
	std::string name_;
	jlm::linkage linkage_;
	std::unique_ptr<jlm::cfg> cfg_;
};

class fctvariable final : public gblvariable {
public:
	virtual
	~fctvariable();

	inline
	fctvariable(function_node * node)
	: gblvariable(node->type(), node->name())
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

/* data node */

class data_node final : public ipgraph_node {
public:
	virtual
	~data_node() noexcept;

private:
	inline
	data_node(
		jlm::ipgraph & clg,
		const std::string & name,
		const jive::type & type,
		const jlm::linkage & linkage,
		bool constant)
	: ipgraph_node(clg)
	, constant_(constant)
	, name_(name)
	, linkage_(linkage)
	, type_(std::move(type.copy()))
	{}

public:
	virtual const jive::type &
	type() const noexcept override;

	const std::string &
	name() const noexcept override;

	virtual const jlm::linkage &
	linkage() const noexcept override;

	inline bool
	constant() const noexcept
	{
		return constant_;
	}

	inline const tacsvector_t &
	initialization() const noexcept
	{
		return init_;
	}

	const jive::valuetype &
	valuetype() const noexcept
	{
		return *static_cast<const jive::valuetype*>(type_.get());
	}

	void
	set_initialization(tacsvector_t init);

	static inline data_node *
	create(
		jlm::ipgraph & clg,
		const std::string & name,
		const jive::valuetype & type,
		const jlm::linkage & linkage,
		bool constant)
	{
		std::unique_ptr<data_node> node(new data_node(clg, name, type, linkage, constant));
		auto ptr = node.get();
		clg.add_node(std::move(node));
		return ptr;
	}

private:
	bool constant_;
	std::string name_;
	tacsvector_t init_;
	jlm::linkage linkage_;
	std::unique_ptr<jive::type> type_;
};

}

#endif

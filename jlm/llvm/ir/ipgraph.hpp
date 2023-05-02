/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_IPGRAPH_HPP
#define JLM_LLVM_IR_IPGRAPH_HPP

#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/ir/variable.hpp>

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

	const ipgraph_node *
	find(const std::string & name) const noexcept;

private:
	std::vector<std::unique_ptr<ipgraph_node>> nodes_;
};

/* clg node */

class output;

class ipgraph_node {
	typedef std::unordered_set<const ipgraph_node*>::const_iterator const_iterator;
public:
	virtual
	~ipgraph_node();

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

	virtual bool
	hasBody() const noexcept = 0;

private:
	jlm::ipgraph & clg_;
	std::unordered_set<const ipgraph_node*> dependencies_;
};

class function_node final : public ipgraph_node {
public:
	virtual
	~function_node();

private:
	inline
	function_node(
		jlm::ipgraph & clg,
		const std::string & name,
		const FunctionType & type,
		const jlm::linkage & linkage,
		const attributeset & attributes)
	: ipgraph_node(clg)
  , FunctionType_(type)
	, name_(name)
	, linkage_(linkage)
	, attributes_(attributes)
	{}

public:
	inline jlm::cfg *
	cfg() const noexcept
	{
		return cfg_.get();
	}

	virtual const jive::type &
	type() const noexcept override;

	const FunctionType &
	fcttype() const noexcept
	{
		return FunctionType_;
	}

	virtual const jlm::linkage &
	linkage() const noexcept override;

	virtual const std::string &
	name() const noexcept override;

	virtual bool
	hasBody() const noexcept override;

	const attributeset &
	attributes() const noexcept
	{
		return attributes_;
	}

	/**
	* \brief Adds \p cfg to the function node. If the function node already has a CFG, then it is
		replaced with \p cfg.
	**/
	void
	add_cfg(std::unique_ptr<jlm::cfg> cfg);

	static inline function_node *
	create(
		jlm::ipgraph & ipg,
		const std::string & name,
		const FunctionType & type,
		const jlm::linkage & linkage,
		const attributeset & attributes)
	{
		std::unique_ptr<function_node> node(new function_node(ipg, name, type, linkage, attributes));
		auto tmp = node.get();
		ipg.add_node(std::move(node));
		return tmp;
	}

	static function_node *
	create(
		jlm::ipgraph & ipg,
		const std::string & name,
		const FunctionType & type,
		const jlm::linkage & linkage)
	{
		return create(ipg, name, type, linkage, {});
	}

private:
  FunctionType FunctionType_;
	std::string name_;
	jlm::linkage linkage_;
	attributeset attributes_;
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

class data_node_init final {
public:
	data_node_init(const variable * value)
	: value_(value)
	{}

	data_node_init(tacsvector_t tacs)
	: tacs_(std::move(tacs))
	{
		if (tacs_.empty())
			throw jlm::error("Initialization cannot be empty.");

		auto & tac = tacs_.back();
		if (tac->nresults() != 1)
			throw jlm::error("Last TAC of initialization needs exactly one result.");

		value_ = tac->result(0);
	}

	data_node_init(const data_node_init&) = delete;

	data_node_init(data_node_init && other)
	: tacs_(std::move(other.tacs_))
	, value_(other.value_)
	{}

	data_node_init &
	operator=(const data_node_init&) = delete;

	data_node_init &
	operator=(data_node_init&&) = delete;

	const variable *
	value() const noexcept
	{
		return value_;
	}

	const tacsvector_t &
	tacs() const noexcept
	{
		return tacs_;
	}

private:
	tacsvector_t tacs_;
	const variable * value_;
};

class data_node final : public ipgraph_node {
public:
	virtual
	~data_node();

private:
	inline
	data_node(
		jlm::ipgraph & clg,
		const std::string & name,
		const jive::valuetype & valueType,
		const jlm::linkage & linkage,
    std::string section,
		bool constant)
	: ipgraph_node(clg)
	, constant_(constant)
	, name_(name)
  , Section_(std::move(section))
	, linkage_(linkage)
  , ValueType_(valueType.copy())
	{}

public:
	virtual const PointerType &
	type() const noexcept override;

  [[nodiscard]] const jive::valuetype &
  GetValueType() const noexcept
  {
    return *AssertedCast<jive::valuetype>(ValueType_.get());
  }

	const std::string &
	name() const noexcept override;

	virtual const jlm::linkage &
	linkage() const noexcept override;

	virtual bool
	hasBody() const noexcept override;

	inline bool
	constant() const noexcept
	{
		return constant_;
	}

  const std::string &
  Section() const noexcept
  {
    return Section_;
  }

	inline const data_node_init *
	initialization() const noexcept
	{
		return init_.get();
	}

	void
	set_initialization(std::unique_ptr<data_node_init> init)
	{
		if (!init)
			return;

		if (init->value()->type() != GetValueType())
			throw jlm::error("Invalid type.");

		init_ = std::move(init);
	}

	static data_node *
	Create(
		jlm::ipgraph & clg,
		const std::string & name,
    const jive::valuetype & valueType,
		const jlm::linkage & linkage,
    std::string section,
		bool constant)
	{
		std::unique_ptr<data_node> node(new data_node(clg, name, valueType, linkage, std::move(section), constant));
		auto ptr = node.get();
		clg.add_node(std::move(node));
		return ptr;
	}

private:
	bool constant_;
	std::string name_;
  std::string Section_;
	jlm::linkage linkage_;
  std::unique_ptr<jive::type> ValueType_;
	std::unique_ptr<data_node_init> init_;
};

}

#endif

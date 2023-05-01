/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */
#ifndef JLM_LLVM_IR_CFG_HPP
#define JLM_LLVM_IR_CFG_HPP

#include <jlm/llvm/ir/attribute.hpp>
#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-node.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/ir/variable.hpp>
#include <jlm/rvsdg/operation.hpp>
#include <jlm/util/common.hpp>

namespace jive {
	class type;
}

namespace jlm {

class clg_node;
class basic_block;
class ipgraph_module;
class tac;

/** \brief Function argument
*/
class argument final : public variable {
public:
	~argument() override;

	argument(
		const std::string & name,
		const jive::type & type,
		const attributeset & attributes)
	: variable(*type.copy(), name)
	, attributes_(attributes)
	{}

	argument(
		const std::string & name,
		const jive::type & type)
	: variable(*type.copy(), name)
	{}

	argument(
		const std::string & name,
		std::unique_ptr<jive::type> type,
		const attributeset & attributes)
	: variable(std::move(type), name)
	, attributes_(attributes)
	{}

	const attributeset &
	attributes() const noexcept
	{
		return attributes_;
	}

	void
	add(const jlm::attribute & attribute)
	{
		attributes_.insert(attribute);
	}

	void
	add(std::unique_ptr<jlm::attribute> attribute)
	{
		attributes_.insert(std::move(attribute));
	}

	static std::unique_ptr<argument>
	create(
		const std::string & name,
		const jive::type & type,
		const attributeset & attributes)
	{
		return std::make_unique<argument>(name, type, attributes);
	}

	static std::unique_ptr<argument>
	create(
		const std::string & name,
		std::unique_ptr<jive::type> type,
		const attributeset & attributes)
	{
		return std::make_unique<argument>(name, std::move(type), attributes);
	}

	static std::unique_ptr<argument>
	create(
		const std::string & name,
		const jive::type & type)
	{
		return create(name, type, {});
	}

private:
	attributeset attributes_;
};

/* cfg entry node */

class entry_node final : public cfg_node {
public:
	virtual
	~entry_node();

	entry_node(jlm::cfg & cfg)
	: cfg_node(cfg)
	{}

	size_t
	narguments() const noexcept
	{
		return arguments_.size();
	}

	const jlm::argument *
	argument(size_t index) const
	{
		JLM_ASSERT(index < narguments());
		return arguments_[index].get();
	}

	jlm::argument *
	append_argument(std::unique_ptr<jlm::argument> arg)
	{
		arguments_.push_back(std::move(arg));
		return arguments_.back().get();
	}

	std::vector<jlm::argument*>
	arguments() const noexcept
	{
		std::vector<jlm::argument*> arguments;
		for (auto & argument : arguments_)
			arguments.push_back(argument.get());

		return arguments;
	}

private:
	std::vector<std::unique_ptr<jlm::argument>> arguments_;
};

/* cfg exit node */

class exit_node final : public cfg_node {
public:
	virtual
	~exit_node();

	exit_node(jlm::cfg & cfg)
	: cfg_node(cfg)
	{}

	size_t
	nresults() const noexcept
	{
		return results_.size();
	}

	const variable *
	result(size_t index) const
	{
		JLM_ASSERT(index < nresults());
		return results_[index];
	}

	inline void
	append_result(const variable * v)
	{
		results_.push_back(v);
	}

	const std::vector<const variable*>
	results() const noexcept
	{
		return results_;
	}

private:
	std::vector<const variable*> results_;
};

/* control flow graph */

class cfg final {
	class iterator final {
	public:
		inline
		iterator(std::unordered_set<std::unique_ptr<basic_block>>::iterator it)
		: it_(it)
		{}

		inline bool
		operator==(const iterator & other) const noexcept
		{
			return it_ == other.it_;
		}

		inline bool
		operator!=(const iterator & other) const noexcept
		{
			return !(*this == other);
		}

		inline const iterator &
		operator++() noexcept
		{
			++it_;
			return *this;
		}

		inline const iterator
		operator++(int) noexcept
		{
			iterator tmp(it_);
			it_++;
			return tmp;
		}

		inline basic_block *
		node() const noexcept
		{
			return it_->get();
		}

		inline basic_block &
		operator*() const noexcept
		{
			return *it_->get();
		}

		inline basic_block *
		operator->() const noexcept
		{
			return node();
		}

	private:
		std::unordered_set<std::unique_ptr<basic_block>>::iterator it_;
	};

	class const_iterator final {
	public:
		inline
		const_iterator(std::unordered_set<std::unique_ptr<basic_block>>::const_iterator it)
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

		inline const basic_block &
		operator*() noexcept
		{
			return *it_->get();
		}

		inline const basic_block *
		operator->() noexcept
		{
			return it_->get();
		}

	private:
		std::unordered_set<std::unique_ptr<basic_block>>::const_iterator it_;
	};

public:
	~cfg() {}

	cfg(ipgraph_module & im);

	cfg(const cfg&) = delete;

	cfg(cfg&&) = delete;

	cfg &
	operator=(const cfg&) = delete;

	cfg &
	operator=(cfg&&) = delete;

public:
	inline const_iterator
	begin() const
	{
		return const_iterator(nodes_.begin());
	}

	inline iterator
	begin()
	{
		return iterator(nodes_.begin());
	}

	inline const_iterator
	end() const
	{
		return const_iterator(nodes_.end());
	}

	inline iterator
	end()
	{
		return iterator(nodes_.end());
	}

	inline jlm::entry_node *
	entry() const noexcept
	{
		return entry_.get();
	}

	inline jlm::exit_node *
	exit() const noexcept
	{
		return exit_.get();
	}

	inline basic_block *
	add_node(std::unique_ptr<basic_block> bb)
	{
		auto tmp = bb.get();
		nodes_.insert(std::move(bb));
		return tmp;
	}

	inline cfg::iterator
	find_node(basic_block * bb)
	{
		std::unique_ptr<basic_block> up(bb);
		auto it = nodes_.find(up);
		up.release();
		return iterator(it);
	}

	static cfg::iterator
	remove_node(cfg::iterator & it);

	static cfg::iterator
	remove_node(basic_block * bb);

	inline size_t
	nnodes() const noexcept
	{
		return nodes_.size();
	}

	inline ipgraph_module &
	module() const noexcept
	{
		return module_;
	}

	FunctionType
	fcttype() const
	{
		std::vector<const jive::type*> arguments;
		for (size_t n = 0; n < entry()->narguments(); n++)
			arguments.push_back(&entry()->argument(n)->type());

		std::vector<const jive::type*> results;
		for (size_t n = 0; n < exit()->nresults(); n++)
			results.push_back(&exit()->result(n)->type());

		return FunctionType(arguments, results);
	}

	static std::unique_ptr<cfg>
	create(ipgraph_module & im)
	{
		return std::unique_ptr<cfg>(new cfg(im));
	}

private:
	ipgraph_module & module_;
	std::unique_ptr<exit_node> exit_;
	std::unique_ptr<entry_node> entry_;
	std::unordered_set<std::unique_ptr<basic_block>> nodes_;
};

std::vector<cfg_node*>
postorder(const jlm::cfg & cfg);

std::vector<cfg_node*>
reverse_postorder(const jlm::cfg & cfg);

/** Order CFG nodes breadth-first
*
* Note, all nodes that are not dominated by the entry node are ignored.
*
* param cfg Control flow graph
*
* return A vector with all CFG nodes ordered breadth-first
*/
std::vector<cfg_node*>
breadth_first(const jlm::cfg & cfg);

size_t
ntacs(const jlm::cfg & cfg);

}

#endif

/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */
#ifndef JLM_IR_CFG_H
#define JLM_IR_CFG_H

#include <jlm/common.hpp>
#include <jlm/IR/cfg_node.hpp>
#include <jlm/IR/variable.hpp>

#include <jive/vsdg/operators.h>

namespace jive {
	class buffer;

namespace base {
	class type;
}
}

namespace jlm {

class clg_node;
class basic_block;
class tac;

class entry_attribute final : public attribute {
public:
	virtual
	~entry_attribute();

	inline
	entry_attribute()
	: attribute()
	{}

	size_t
	narguments() const noexcept
	{
		return arguments_.size();
	}

	std::shared_ptr<const variable>
	argument(size_t index) const
	{
		JLM_DEBUG_ASSERT(index < narguments());
		return arguments_[index];
	}

	inline void
	append_argument(const std::shared_ptr<const variable> & v)
	{
		return arguments_.push_back(v);
	}

	virtual std::string
	debug_string() const noexcept override;

	virtual std::unique_ptr<attribute>
	copy() const override;

private:
	std::vector<std::shared_ptr<const variable>> arguments_;
};

static inline bool
is_entry_node(const jlm::cfg_node * node)
{
	return dynamic_cast<const jlm::entry_attribute*>(&node->attribute()) != nullptr;
}

class exit_attribute final : public attribute {
public:
	virtual
	~exit_attribute();

	inline
	exit_attribute()
	: attribute()
	{}

	size_t
	nresults() const noexcept
	{
		return results_.size();
	}

	std::shared_ptr<const variable>
	result(size_t index) const
	{
		JLM_DEBUG_ASSERT(index < nresults());
		return results_[index];
	}

	inline void
	append_result(const std::shared_ptr<const variable> & result)
	{
		results_.push_back(result);
	}

	virtual std::string
	debug_string() const noexcept override;

	virtual std::unique_ptr<attribute>
	copy() const override;

private:
	std::vector<std::shared_ptr<const variable>> results_;
};

static inline bool
is_exit_node(const jlm::cfg_node * node)
{
	return dynamic_cast<const jlm::exit_attribute*>(&node->attribute()) != nullptr;
}

class cfg final {
	class iterator final {
	public:
		inline
		iterator(std::unordered_set<std::unique_ptr<cfg_node>>::iterator it)
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

		inline cfg_node &
		operator*() noexcept
		{
			return *it_->get();
		}

		inline cfg_node *
		operator->() noexcept
		{
			return it_->get();
		}

	private:
		std::unordered_set<std::unique_ptr<cfg_node>>::iterator it_;
	};

	class const_iterator final {
	public:
		inline
		const_iterator(std::unordered_set<std::unique_ptr<cfg_node>>::const_iterator it)
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

		inline const cfg_node &
		operator*() noexcept
		{
			return *it_->get();
		}

		inline const cfg_node *
		operator->() noexcept
		{
			return it_->get();
		}

	private:
		std::unordered_set<std::unique_ptr<cfg_node>>::const_iterator it_;
	};

public:
	~cfg() {}

	cfg();

private:
	cfg(const cfg & c);

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

	std::vector<std::unordered_set<cfg_node*>> find_sccs() const;

	void convert_to_dot(jive::buffer & buffer) const;

	bool is_valid() const;

	bool is_closed() const noexcept;

	bool is_linear() const noexcept;

	bool is_acyclic() const;

	bool is_structured() const;

	bool is_reducible() const;

	void prune();

	inline jlm::cfg_node *
	entry() const noexcept
	{
		return entry_;
	}

	inline jlm::cfg_node *
	exit() const noexcept
	{
		return exit_;
	}

	cfg_node *
	create_node(const attribute & attr);

	inline size_t
	nnodes() const noexcept
	{
		return nodes_.size();
	}

	inline std::shared_ptr<variable>
	append_argument(const std::string & name, const jive::base::type & type)
	{
		std::shared_ptr<variable> v(new variable(type, name));
		static_cast<entry_attribute*>(&entry()->attribute())->append_argument(v);
		return v;
	}

	inline size_t
	narguments() const noexcept
	{
		return static_cast<entry_attribute*>(&entry()->attribute())->narguments();
	}

	inline std::shared_ptr<const variable>
	argument(size_t index) const
	{
		return static_cast<entry_attribute*>(&entry()->attribute())->argument(index);
	}

	inline void
	append_result(const std::shared_ptr<const variable> & result)
	{
		static_cast<exit_attribute*>(&exit()->attribute())->append_result(result);
	}

	inline size_t
	nresults() const noexcept
	{
		return static_cast<exit_attribute*>(&exit()->attribute())->nresults();
	}

	inline std::shared_ptr<const variable>
	result(size_t index) const
	{
		return static_cast<exit_attribute*>(&exit()->attribute())->result(index);
	}

private:
	void remove_node(cfg_node * node);

	cfg_node * entry_;
	cfg_node * exit_;
	std::unordered_set<std::unique_ptr<cfg_node>> nodes_;
};

static inline std::vector<std::shared_ptr<const jlm::variable>>
create_variables(jlm::cfg & cfg, const jive::operation & operation)
{
	std::vector<std::shared_ptr<const variable>> results;
	for (size_t n = 0; n < operation.nresults(); n++) {
		std::shared_ptr<const jlm::variable> v(new jlm::variable(operation.result_type(n), ""));
		results.push_back(v);
	}

	return results;
}

}

void
jive_cfg_view(const jlm::cfg & self);

#endif

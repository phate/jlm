/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */
#ifndef JLM_IR_CFG_H
#define JLM_IR_CFG_H

#include <jlm/common.hpp>
#include <jlm/IR/cfg_node.hpp>
#include <jlm/IR/variable.hpp>

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

	const variable *
	argument(size_t index) const
	{
		JLM_DEBUG_ASSERT(index < narguments());
		return arguments_[index];
	}

	inline void
	append_argument(const variable * v)
	{
		return arguments_.push_back(v);
	}

	virtual std::string
	debug_string() const noexcept override;

	virtual std::unique_ptr<attribute>
	copy() const override;

private:
	std::vector<const variable*> arguments_;
};

class cfg final {
	class exit_node;
public:
	~cfg() {}

	cfg();

	cfg(clg_node  & clg_node);

private:
	cfg(const cfg & c);

public:
	std::vector<std::unordered_set<cfg_node*>> find_sccs() const;

	void convert_to_dot(jive::buffer & buffer) const;

	bool is_valid() const;

	bool is_closed() const noexcept;

	bool is_linear() const noexcept;

	bool is_acyclic() const;

	bool is_structured() const;

	bool is_reducible() const;

	void prune();

	void
	destruct_ssa();

	inline jlm::cfg_node *
	entry() const noexcept
	{
		return entry_;
	}

	inline jlm::cfg::exit_node * exit() const noexcept { return exit_; }

	inline jlm::clg_node * function() const noexcept { return clg_node_; }

	cfg_node *
	create_node(const attribute & attr);

	jlm::variable *
	create_variable(const jive::base::type & type);

	jlm::variable *
	create_variable(const jive::base::type & type, const std::string & name);

	inline size_t
	nnodes() const noexcept
	{
		return nodes_.size();
	}

	inline variable *
	append_argument(const std::string & name, const jive::base::type & type)
	{
		auto v = create_variable(type, name);
		static_cast<entry_attribute*>(&entry()->attribute())->append_argument(v);
		return v;
	}

	inline size_t
	narguments() const noexcept
	{
		return static_cast<entry_attribute*>(&entry()->attribute())->narguments();
	}

	inline const variable *
	argument(size_t index) const
	{
		return static_cast<entry_attribute*>(&entry()->attribute())->argument(index);
	}

	inline void
	append_result(variable * result)
	{
		exit_->append_result(result);
	}

	inline size_t
	nresults() const noexcept
	{
		return exit_->nresults();
	}

	inline const variable *
	result(size_t index) const
	{
		return exit_->result(index);
	}

private:
	class exit_node final : public cfg_node {
	public:
		virtual ~exit_node() noexcept;

		exit_node(jlm::cfg & cfg) noexcept;

		virtual std::string debug_string() const override;

		inline void
		append_result(variable * result)
		{
			results_.push_back(result);
		}

		inline size_t
		nresults() const noexcept
		{
			return results_.size();
		}

		inline const variable *
		result(size_t index) const
		{
			JLM_DEBUG_ASSERT(index < results_.size());
			return results_[index];
		}

	private:
		std::vector<variable*> results_;
	};

	void remove_node(cfg_node * node);
	void create_exit_node();

	cfg_node * entry_;
	cfg::exit_node * exit_;
	jlm::clg_node * clg_node_;
	std::unordered_set<std::unique_ptr<cfg_node>> nodes_;
	std::unordered_set<std::unique_ptr<variable>> variables_;
};

}

void
jive_cfg_view(const jlm::cfg & self);

#endif

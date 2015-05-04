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

class cfg final {
	class enter_node;
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

	inline jlm::cfg::enter_node * enter() const noexcept { return enter_; }
	inline jlm::cfg::exit_node * exit() const noexcept { return exit_; }

	inline jlm::clg_node * function() const noexcept { return clg_node_; }

	basic_block * create_basic_block();

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
		return enter_->append_argument(name, type);
	}

	inline size_t
	narguments() const noexcept
	{
		return enter_->narguments();
	}

	inline const std::string &
	argument_name(size_t index) const
	{
		return enter_->argument_name(index);
	}

	inline const jive::base::type &
	argument_type(size_t index) const
	{
		return enter_->argument_type(index);
	}

	inline const variable *
	argument(size_t index) const
	{
		return enter_->argument(index);
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
	class enter_node final : public cfg_node {
	public:
		virtual ~enter_node() noexcept;

		enter_node(jlm::cfg & cfg) noexcept;

		virtual std::string debug_string() const override;

		inline variable *
		append_argument(const std::string & name, const jive::base::type & type)
		{
			arguments_.push_back(cfg()->create_variable(type, name));
			return arguments_[arguments_.size()-1];
		}

		size_t
		narguments() const noexcept
		{
			return arguments_.size();
		}

		const std::string &
		argument_name(size_t index) const;

		const jive::base::type &
		argument_type(size_t index) const;

		const variable *
		argument(size_t index) const;

	private:
		std::vector<variable*> arguments_;
	};

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
	void create_enter_node();
	void create_exit_node();

	cfg::enter_node * enter_;
	cfg::exit_node * exit_;
	jlm::clg_node * clg_node_;
	std::unordered_set<std::unique_ptr<cfg_node>> nodes_;
	std::unordered_set<std::unique_ptr<variable>> variables_;
};

}

void
jive_cfg_view(const jlm::cfg & self);

#endif

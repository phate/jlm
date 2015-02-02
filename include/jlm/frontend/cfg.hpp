/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_FRONTEND_CFG_H
#define JLM_FRONTEND_CFG_H

#include <jlm/common.hpp>
#include <jlm/frontend/cfg_node.hpp>

namespace jive {
	class buffer;

namespace base {
	class type;
}
}

namespace jlm {
namespace frontend {
	class clg_node;
	class basic_block;
	class output;
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

	inline jlm::frontend::cfg::enter_node * enter() const noexcept { return enter_; }
	inline jlm::frontend::cfg::exit_node * exit() const noexcept { return exit_; }

	inline jlm::frontend::clg_node * function() const noexcept { return clg_node_; }

	basic_block * create_basic_block();

	inline size_t
	nnodes() const noexcept
	{
		return nodes_.size();
	}

	inline const output *
	append_argument(const std::string & name, const jive::base::type & type)
	{
		return enter_->append_argument(name, type);
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

	inline const output *
	argument(size_t index) const
	{
		return enter_->argument(index);
	}

	inline void
	append_result(const output * result)
	{
		exit_->append_result(result);
	}

	inline const output *
	result(size_t index) const
	{
		return exit_->result(index);
	}

private:
	class enter_node final : public cfg_node {
	public:
		virtual ~enter_node() noexcept;

		enter_node(jlm::frontend::cfg & cfg) noexcept;

		virtual std::string debug_string() const override;

		const output *
		append_argument(const std::string & name, const jive::base::type & type);

		const std::string &
		argument_name(size_t index) const;

		const jive::base::type &
		argument_type(size_t index) const;

		const output *
		argument(size_t index) const;

	private:
		std::vector<std::unique_ptr<tac>> arguments_;
	};

	class exit_node final : public cfg_node {
	public:
		virtual ~exit_node() noexcept;

		exit_node(jlm::frontend::cfg & cfg) noexcept;

		virtual std::string debug_string() const override;

		inline void
		append_result(const output * result)
		{
			results_.push_back(result);
		}

		inline const output *
		result(size_t index) const
		{
			JLM_DEBUG_ASSERT(index < results_.size());
			return results_[index];
		}

	private:
		std::vector<const output*> results_;
	};

	void remove_node(cfg_node * node);
	void create_enter_node();
	void create_exit_node();

	cfg::enter_node * enter_;
	cfg::exit_node * exit_;
	jlm::frontend::clg_node * clg_node_;
	std::unordered_set<std::unique_ptr<cfg_node>> nodes_;
};

}
}

void
jive_cfg_view(const jlm::frontend::cfg & self);

#endif

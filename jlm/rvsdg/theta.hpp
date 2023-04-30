/*
 * Copyright 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_THETA_HPP
#define JLM_RVSDG_THETA_HPP

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/structural-node.hpp>

namespace jive {

/* theta operation */

class theta_op final : public structural_op {
public:
	virtual
	~theta_op() noexcept;
	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;
};

/* theta node */

class theta_input;
class theta_output;

class theta_node final : public structural_node {
public:
	class loopvar_iterator {
	public:
		inline constexpr
		loopvar_iterator(jive::theta_output * output) noexcept
		: output_(output)
		{}

		const loopvar_iterator &
		operator++() noexcept;

		inline const loopvar_iterator
		operator++(int) noexcept
		{
			loopvar_iterator it(*this);
			++(*this);
			return it;
		}

		inline bool
		operator==(const loopvar_iterator & other) const noexcept
		{
			return output_ == other.output_;
		}

		inline bool
		operator!=(const loopvar_iterator & other) const noexcept
		{
			return !(*this == other);
		}

		inline theta_output *
		operator*() noexcept
		{
			return output_;
		}

		inline theta_output **
		operator->() noexcept
		{
			return &output_;
		}

		inline jive::theta_output *
		output() const noexcept
		{
			return output_;
		}

	private:
		jive::theta_output * output_;
	};

	virtual
	~theta_node();

private:
	inline
	theta_node(jive::region * parent)
	: structural_node(jive::theta_op(), parent, 1)
	{
		auto predicate = jive_control_false(subregion());
		result::create(subregion(), predicate, nullptr, ctltype(2));
	}

public:
	static jive::theta_node *
	create(jive::region * parent)
	{
		return new jive::theta_node(parent);
	}

	inline jive::region *
	subregion() const noexcept
	{
		return structural_node::subregion(0);
	}

	inline jive::result *
	predicate() const noexcept
	{
		auto result = subregion()->result(0);
		JIVE_DEBUG_ASSERT(dynamic_cast<const ctltype*>(&result->type()));
		return result;
	}

	inline void
	set_predicate(jive::output * p)
	{
		auto node = node_output::node(predicate()->origin());

		predicate()->divert_to(p);
		if (node && !node->has_users())
			remove(node);
	}

	inline size_t
	nloopvars() const noexcept
	{
		JIVE_DEBUG_ASSERT(ninputs() == noutputs());
		return ninputs();
	}

	inline theta_node::loopvar_iterator
	begin() const
	{
		if (ninputs() == 0)
			return loopvar_iterator(nullptr);

		return loopvar_iterator(output(0));
	}

	inline theta_node::loopvar_iterator
	end() const
	{
		return loopvar_iterator(nullptr);
	}

	theta_input *
	input(size_t index) const noexcept;

	theta_output *
	output(size_t index) const noexcept;

	jive::theta_output *
	add_loopvar(jive::output * origin);

	virtual jive::theta_node *
	copy(jive::region * region, jive::substitution_map & smap) const override;
};

/* theta input */

class theta_input final : public structural_input {
	friend theta_node;
	friend theta_output;
public:
	virtual
	~theta_input() noexcept;

private:
	inline
	theta_input(
		theta_node * node,
		jive::output * origin,
		const jive::port & port)
	: structural_input(node, origin, port)
	, output_(nullptr)
	{}

public:
	theta_node *
	node() const noexcept
	{
		return static_cast<theta_node*>(structural_input::node());
	}

	inline jive::theta_output *
	output() const noexcept
	{
		return output_;
	}

	inline jive::argument *
	argument() const noexcept
	{
		JIVE_DEBUG_ASSERT(arguments.size() == 1);
		return arguments.first();
	}

	jive::result *
	result() const noexcept;

private:
	jive::theta_output * output_;
};

static inline bool
is_theta_input(const jive::input * input) noexcept
{
	return dynamic_cast<const jive::theta_input*>(input) != nullptr;
}

static inline bool
is_invariant(const jive::theta_input * input) noexcept
{
	return input->result()->origin() == input->argument();
}

/* theta output */

class theta_output final : public structural_output {
	friend theta_node;
	friend theta_input;
public:
	virtual
	~theta_output() noexcept;

private:
	inline
	theta_output(
		theta_node * node,
		const jive::port & port)
	: structural_output(node, port)
	, input_(nullptr)
	{}

public:
	theta_node *
	node() const noexcept
	{
		return static_cast<theta_node*>(structural_output::node());
	}

	inline jive::theta_input *
	input() const noexcept
	{
		return input_;
	}

	inline jive::argument *
	argument() const noexcept
	{
		return input_->argument();
	}

	inline jive::result *
	result() const noexcept
	{
		JIVE_DEBUG_ASSERT(results.size() == 1);
		return results.first();
	}

private:
	jive::theta_input * input_;
};

static inline bool
is_theta_output(const jive::theta_output * output) noexcept
{
	return dynamic_cast<const jive::theta_output*>(output) != nullptr;
}

static inline bool
is_invariant(const jive::theta_output * output) noexcept
{
	return output->result()->origin() == output->argument();
}

/* theta node method definitions */

inline jive::theta_input *
theta_node::input(size_t index) const noexcept
{
	return static_cast<theta_input*>(node::input(index));
}

inline jive::theta_output *
theta_node::output(size_t index) const noexcept
{
	return static_cast<theta_output*>(node::output(index));
}

/* theta input method definitions */

inline jive::result *
theta_input::result() const noexcept
{
	return output_->result();
}

}

#endif

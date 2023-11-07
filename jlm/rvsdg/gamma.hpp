/*
 * Copyright 2010 2011 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_GAMMA_HPP
#define JLM_RVSDG_GAMMA_HPP

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/structural-normal-form.hpp>

namespace jlm::rvsdg
{

/* gamma normal form */

class gamma_normal_form final : public structural_normal_form {
public:
	virtual
	~gamma_normal_form() noexcept;

	gamma_normal_form(
		const std::type_info & operator_class,
		jlm::rvsdg::node_normal_form * parent,
		jlm::rvsdg::graph * graph) noexcept;

	virtual bool
	normalize_node(jlm::rvsdg::node * node) const override;

	virtual void
	set_predicate_reduction(bool enable);

	inline bool
	get_predicate_reduction() const noexcept
	{
		return enable_predicate_reduction_;
	}

	virtual void
	set_invariant_reduction(bool enable);

	inline bool
	get_invariant_reduction() const noexcept
	{
		return enable_invariant_reduction_;
	}

	virtual void
	set_control_constant_reduction(bool enable);

	inline bool
	get_control_constant_reduction() const noexcept
	{
		return enable_control_constant_reduction_;
	}

private:
	bool enable_predicate_reduction_;
	bool enable_invariant_reduction_;
	bool enable_control_constant_reduction_;
};

/* gamma operation */

class output;
class type;

class gamma_op final : public structural_op {
public:
	virtual
	~gamma_op() noexcept;

	inline constexpr
	gamma_op(size_t nalternatives) noexcept
	: structural_op()
	, nalternatives_(nalternatives)
	{}

	inline size_t
	nalternatives() const noexcept
	{
		return nalternatives_;
	}

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jlm::rvsdg::operation>
	copy() const override;

	virtual bool
	operator==(const operation & other) const noexcept override;

	static jlm::rvsdg::gamma_normal_form *
	normal_form(jlm::rvsdg::graph * graph) noexcept
	{
		return static_cast<jlm::rvsdg::gamma_normal_form*>(graph->node_normal_form(typeid(gamma_op)));
	}

private:
	size_t nalternatives_;
};

/* gamma node */

class gamma_input;
class gamma_output;

class gamma_node : public jlm::rvsdg::structural_node {
public:
	virtual
	~gamma_node();

private:
	gamma_node(jlm::rvsdg::output * predicate, size_t nalternatives);

	class entryvar_iterator {
	public:
		inline constexpr
		entryvar_iterator(jlm::rvsdg::gamma_input * input) noexcept
		: input_(input)
		{}

		inline jlm::rvsdg::gamma_input *
		input() const noexcept
		{
			return input_;
		}

		const entryvar_iterator &
		operator++() noexcept;

		inline const entryvar_iterator
		operator++(int) noexcept
		{
			entryvar_iterator it(*this);
			++(*this);
			return it;
		}

		inline bool
		operator==(const entryvar_iterator & other) const noexcept
		{
			return input_ == other.input_;
		}

		inline bool
		operator!=(const entryvar_iterator & other) const noexcept
		{
			return !(*this == other);
		}

		inline jlm::rvsdg::gamma_input &
		operator*() noexcept
		{
			return *input_;
		}

		inline jlm::rvsdg::gamma_input *
		operator->() noexcept
		{
			return input_;
		}

	private:
		jlm::rvsdg::gamma_input * input_;
	};

	class exitvar_iterator {
	public:
		inline constexpr
		exitvar_iterator(jlm::rvsdg::gamma_output * output) noexcept
		: output_(output)
		{}

		inline jlm::rvsdg::gamma_output *
		output() const noexcept
		{
			return output_;
		}

		const exitvar_iterator &
		operator++() noexcept;

		inline const exitvar_iterator
		operator++(int) noexcept
		{
			exitvar_iterator it(*this);
			++(*this);
			return it;
		}

		inline bool
		operator==(const exitvar_iterator & other) const noexcept
		{
			return output_ == other.output_;
		}

		inline bool
		operator!=(const exitvar_iterator & other) const noexcept
		{
			return !(*this == other);
		}

		inline gamma_output &
		operator*() noexcept
		{
			return *output_;
		}

		inline gamma_output *
		operator->() noexcept
		{
			return output_;
		}

	private:
		jlm::rvsdg::gamma_output * output_;
	};

public:
	static jlm::rvsdg::gamma_node *
	create(jlm::rvsdg::output * predicate, size_t nalternatives)
	{
		return new jlm::rvsdg::gamma_node(predicate, nalternatives);
	}

	jlm::rvsdg::gamma_input *
	predicate() const noexcept;

	inline size_t
	nentryvars() const noexcept
	{
		JLM_ASSERT(node::ninputs() != 0);
		return node::ninputs()-1;
	}

	inline size_t
	nexitvars() const noexcept
	{
		return node::noutputs();
	}

	jlm::rvsdg::gamma_input *
	entryvar(size_t index) const noexcept;

	jlm::rvsdg::gamma_output *
	exitvar(size_t index) const noexcept;

	inline gamma_node::entryvar_iterator
	begin_entryvar() const
	{
		if (nentryvars() == 0)
			return entryvar_iterator(nullptr);

		return entryvar_iterator(entryvar(0));
	}

	inline gamma_node::entryvar_iterator
	end_entryvar() const
	{
		return entryvar_iterator(nullptr);
	}

	inline gamma_node::exitvar_iterator
	begin_exitvar() const
	{
		if (nexitvars() == 0)
			return exitvar_iterator(nullptr);

		return exitvar_iterator(exitvar(0));
	}

	inline gamma_node::exitvar_iterator
	end_exitvar() const
	{
		return exitvar_iterator(nullptr);
	}

	jlm::rvsdg::gamma_input *
	add_entryvar(jlm::rvsdg::output * origin);

	jlm::rvsdg::gamma_output *
	add_exitvar(const std::vector<jlm::rvsdg::output*> & values);

  /**
   * Removes all gamma outputs and their respective results. The outputs must have no users and match the condition
   * specified by \p match.
   *
   * @tparam F A type that supports the function call operator: bool operator(const gamma_output&)
   * @param match Defines the condition of the elements to remove.
   */
  template<typename F> void
  RemoveGammaOutputsWhere(const F& match);

  /**
   * Removes all outputs that have no users.
   */
  void
  PruneOutputs()
  {
    auto match = [](const gamma_output&)
    {
      return true;
    };

    RemoveGammaOutputsWhere(match);
  }

	virtual jlm::rvsdg::gamma_node *
	copy(jlm::rvsdg::region * region, jlm::rvsdg::substitution_map & smap) const override;
};

/* gamma input */

class gamma_input final : public structural_input {
	friend gamma_node;
public:
	virtual
	~gamma_input() noexcept;

private:
	inline
	gamma_input(
		gamma_node * node,
		jlm::rvsdg::output * origin,
		const jlm::rvsdg::port & port)
	: structural_input(node, origin, port)
	{}

public:
	gamma_node *
	node() const noexcept
	{
		return static_cast<gamma_node*>(structural_input::node());
	}

	inline argument_list::iterator
	begin()
	{
		return arguments.begin();
	}

	inline argument_list::const_iterator
	begin() const
	{
		return arguments.begin();
	}

	inline argument_list::iterator
	end()
	{
		return arguments.end();
	}

	inline argument_list::const_iterator
	end() const
	{
		return arguments.end();
	}

	inline size_t
	narguments() const noexcept
	{
		return arguments.size();
	}

	inline jlm::rvsdg::argument *
	argument(size_t n) const noexcept
	{
		JLM_ASSERT(n < narguments());
		auto argument = node()->subregion(n)->argument(index()-1);
		JLM_ASSERT(argument->input() == this);
		return argument;
	}
};

static inline bool
is_gamma_input(const jlm::rvsdg::input * input) noexcept
{
	return dynamic_cast<const jlm::rvsdg::gamma_input*>(input) != nullptr;
}

/* gamma output */

class gamma_output final : public structural_output {
	friend gamma_node;
public:
	virtual
	~gamma_output() noexcept;

private:
	inline
	gamma_output(
		gamma_node * node,
		const jlm::rvsdg::port & port)
	: structural_output(node, port)
	{}

public:
	gamma_node *
	node() const noexcept
	{
		return static_cast<gamma_node*>(structural_output::node());
	}

	inline result_list::iterator
	begin()
	{
		return results.begin();
	}

	inline result_list::const_iterator
	begin() const
	{
		return results.begin();
	}

	inline result_list::iterator
	end()
	{
		return results.end();
	}

	inline result_list::const_iterator
	end() const
	{
		return results.end();
	}

	inline size_t
	nresults() const noexcept
	{
		return results.size();
	}

	inline jlm::rvsdg::result *
	result(size_t n) const noexcept
	{
		JLM_ASSERT(n < nresults());
		auto result = node()->subregion(n)->result(index());
		JLM_ASSERT(result->output() == this);
		return result;
	}
};

static inline bool
is_gamma_output(const jlm::rvsdg::input * input) noexcept
{
	return dynamic_cast<const jlm::rvsdg::gamma_input*>(input) != nullptr;
}

/* gamma node method definitions */

inline
gamma_node::gamma_node(jlm::rvsdg::output * predicate, size_t nalternatives)
: structural_node(jlm::rvsdg::gamma_op(nalternatives), predicate->region(), nalternatives)
{
	node::add_input(std::unique_ptr<node_input>(
		new gamma_input(this, predicate, ctltype(nalternatives))));
}
inline jlm::rvsdg::gamma_input *
gamma_node::predicate() const noexcept
{
	return static_cast<jlm::rvsdg::gamma_input*>(structural_node::input(0));
}

inline jlm::rvsdg::gamma_input *
gamma_node::entryvar(size_t index) const noexcept
{
	return static_cast<gamma_input*>(node::input(index+1));
}

inline jlm::rvsdg::gamma_output *
gamma_node::exitvar(size_t index) const noexcept
{
	return static_cast<gamma_output*>(node::output(index));
}

inline jlm::rvsdg::gamma_input *
gamma_node::add_entryvar(jlm::rvsdg::output * origin)
{
	node::add_input(std::unique_ptr<node_input>(
		new gamma_input(this, origin, origin->type())));

	for (size_t n = 0; n < nsubregions(); n++)
		argument::create(subregion(n), input(ninputs()-1), origin->type());

	return static_cast<jlm::rvsdg::gamma_input*>(input(ninputs()-1));
}

inline jlm::rvsdg::gamma_output *
gamma_node::add_exitvar(const std::vector<jlm::rvsdg::output*> & values)
{
	if (values.size() != nsubregions())
		throw jlm::util::error("Incorrect number of values.");

	const auto & port = values[0]->port();
	node::add_output(std::unique_ptr<node_output>(new gamma_output(this, port)));

	auto output = exitvar(nexitvars()-1);
	for (size_t n = 0; n < nsubregions(); n++)
		result::create(subregion(n), values[n], output, port);

	return output;
}

template<typename F> void
gamma_node::RemoveGammaOutputsWhere(const F& match)
{
  // iterate backwards to avoid the invalidation of 'n' by RemoveOutput()
  for (size_t n = noutputs()-1; n != static_cast<size_t>(-1); n--)
  {
    auto & gammaOutput = *util::AssertedCast<const gamma_output>(output(n));
    if (gammaOutput.nusers() == 0 && match(gammaOutput))
    {
      for (size_t r = 0; r < nsubregions(); r++)
      {
        subregion(r)->RemoveResult(n);
      }

      RemoveOutput(n);
    }
  }
}

}

#endif

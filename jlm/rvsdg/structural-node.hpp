/*
 * Copyright 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_STRUCTURAL_NODE_HPP
#define JLM_RVSDG_STRUCTURAL_NODE_HPP

#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jive {

/* structural node */

class structural_input;
class structural_op;
class structural_output;

class structural_node : public node {
public:
	virtual
	~structural_node();

protected:
	structural_node(
		/* FIXME: use move semantics instead of copy semantics for op */
		const jive::structural_op & op,
		jive::region * region,
		size_t nsubregions);

public:
	inline size_t
	nsubregions() const noexcept
	{
		return subregions_.size();
	}

	inline jive::region *
	subregion(size_t index) const noexcept
	{
		JIVE_DEBUG_ASSERT(index < nsubregions());
		return subregions_[index].get();
	}

	inline jive::structural_input *
	input(size_t index) const noexcept;

	inline jive::structural_output *
	output(size_t index) const noexcept;

	structural_input *
	append_input(std::unique_ptr<structural_input> input);

	structural_output *
	append_output(std::unique_ptr<structural_output> output);

	inline void
	remove_input(size_t index)
	{
		node::remove_input(index);
	}

	inline void
	remove_output(size_t index)
	{
		node::remove_output(index);
	}

private:
	std::vector<std::unique_ptr<jive::region>> subregions_;
};

/* structural input class */

typedef jive::detail::intrusive_list<
	jive::argument,
	jive::argument::structural_input_accessor
> argument_list;

class structural_input : public node_input {
	friend structural_node;
public:
	virtual
	~structural_input() noexcept;

protected:
	structural_input(
		jive::structural_node * node,
		jive::output * origin,
		const jive::port & port);

public:
	static structural_input *
	create(
		structural_node * node,
		jive::output * origin,
		const jive::port & port)
	{
		auto input = std::unique_ptr<structural_input>(new structural_input(node, origin, port));
		return node->append_input(std::move(input));
	}

	structural_node *
	node() const noexcept
	{
		return static_cast<structural_node*>(node_input::node());
	}

	argument_list arguments;
};

/* structural output class */

typedef jive::detail::intrusive_list<
	jive::result,
	jive::result::structural_output_accessor
> result_list;

class structural_output : public node_output {
	friend structural_node;

public:
	virtual
	~structural_output() noexcept;

protected:
	structural_output(
		jive::structural_node * node,
		const jive::port & port);

public:
	static structural_output *
	create(
		structural_node * node,
		const jive::port & port)
	{
		auto output = std::unique_ptr<structural_output>(new structural_output(node, port));
		return node->append_output(std::move(output));
	}

	structural_node *
	node() const noexcept
	{
		return static_cast<structural_node*>(node_output::node());
	}

	result_list results;
};

/* structural node method definitions */

inline jive::structural_input *
structural_node::input(size_t index) const noexcept
{
	return static_cast<structural_input*>(node::input(index));
}

inline jive::structural_output *
structural_node::output(size_t index) const noexcept
{
	return static_cast<structural_output*>(node::output(index));
}

template <class Operation> bool
region::Contains(const jive::region & region, bool checkSubregions)
{
  for (auto & node : region.nodes)
  {
    if (is<Operation>(&node))
    {
      return true;
    }

    if (!checkSubregions)
    {
      continue;
    }

    if (auto structuralNode = dynamic_cast<const jive::structural_node*>(&node))
    {
      for (size_t n = 0; n < structuralNode->nsubregions(); n++)
      {
        if (Contains<Operation>(*structuralNode->subregion(n), checkSubregions))
        {
          return true;
        }
      }
    }
  }

  return false;
}

}

#endif

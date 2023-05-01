/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_NODE_HPP
#define JLM_RVSDG_NODE_HPP

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <unordered_set>
#include <utility>

#include <jlm/rvsdg/operation.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/intrusive-list.hpp>
#include <jlm/util/strfmt.hpp>

namespace jive {
namespace base {
	class type;
}

class graph;
class node_normal_form;
class output;
class substitution_map;

/* inputs */

class input {
	friend jive::node;
	friend jive::region;

public:
	virtual
	~input() noexcept;

	input(
		jive::output * origin,
		jive::region * region,
		const jive::port & port);

	input(const input &) = delete;

	input(input &&) = delete;

	input &
	operator=(const input &) = delete;

	input &
	operator=(input &&) = delete;

	inline size_t
	index() const noexcept
	{
		return index_;
	}

	jive::output *
	origin() const noexcept
	{
		return origin_;
	}

	void
	divert_to(jive::output * new_origin);

	inline const jive::type &
	type() const noexcept
	{
		return port_->type();
	}

	inline jive::region *
	region() const noexcept
	{
		return region_;
	}

	inline const jive::port &
	port() const noexcept
	{
		return *port_;
	}

	virtual std::string
	debug_string() const;

	inline void
	replace(const jive::port & port)
	{
		if (port_->type() != port.type())
			throw type_error(port_->type().debug_string(), port.type().debug_string());

		port_ = port.copy();
	}

  /**
   * Retrieve the associated node from \p input if \p input is derived from jive::node_input.
   *
   * @param input The input from which to retrieve the node.
   * @return The node associated with \p input if input is derived from jive::node_input, otherwise nullptr.
   */
  [[nodiscard]] static jive::node *
  GetNode(const jive::input & input) noexcept;

  template <class T>
  class iterator
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T*;
    using difference_type = std::ptrdiff_t;
    using pointer = T**;
    using reference = T*&;

		static_assert(std::is_base_of<jive::input, T>::value,
			"Template parameter T must be derived from jive::input.");

	protected:
		constexpr
		iterator(T * value)
		: value_(value)
		{}

		virtual T *
		next() const
		{
			/*
				I cannot make this method abstract due to the return value of operator++(int).
				This is the best I could come up with as a workaround.
			*/
			throw compiler_error("This method must be overloaded.");
		}

	public:
		T *
		value() const noexcept
		{
			return value_;
		}

		T &
		operator*()
		{
			JIVE_DEBUG_ASSERT(value_ != nullptr);
			return *value_;
		}

		T *
		operator->() const
		{
			return value_;
		}

		iterator<T> &
		operator++()
		{
			value_ = next();
			return *this;
		}

		iterator<T>
		operator++(int)
		{
			iterator<T> tmp = *this;
			++*this;
			return tmp;
		}

		virtual bool
		operator==(const iterator<T> & other) const
		{
			return value_ == other.value_;
		}

		bool
		operator!=(const iterator<T> & other) const
		{
			return !operator==(other);
		}

	private:
		T * value_;
	};

	template <class T>
	class constiterator
	{
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const T*;
    using difference_type = std::ptrdiff_t;
    using pointer = const T**;
    using reference = const T*&;

		static_assert(std::is_base_of<jive::input, T>::value,
			"Template parameter T must be derived from jive::input.");

	protected:
		constexpr
		constiterator(const T * value)
		: value_(value)
		{}

		virtual const T *
		next() const
		{
			/*
				I cannot make this method abstract due to the return value of operator++(int).
				This is the best I could come up with as a workaround.
			*/
			throw compiler_error("This method must be overloaded.");
		}

	public:
		const T *
		value() const noexcept
		{
			return value_;
		}

		const T &
		operator*()
		{
			JIVE_DEBUG_ASSERT(value_ != nullptr);
			return *value_;
		}

		const T *
		operator->() const
		{
			return value_;
		}

		constiterator<T> &
		operator++()
		{
			value_ = next();
			return *this;
		}

		constiterator<T>
		operator++(int)
		{
			constiterator<T> tmp = *this;
			++*this;
			return tmp;
		}

		virtual bool
		operator==(const constiterator<T> & other) const
		{
			return value_ == other.value_;
		}

		bool
		operator!=(const constiterator<T> & other) const
		{
			return !operator==(other);
		}

	private:
		const T * value_;
	};

private:
	size_t index_;
	jive::output * origin_;
	jive::region * region_;
	std::unique_ptr<jive::port> port_;
};

template <class T> static inline bool
is(const jive::input & input) noexcept
{
	static_assert(std::is_base_of<jive::input, T>::value,
		"Template parameter T must be derived from jive::input.");

	return dynamic_cast<const T*>(&input) != nullptr;
}

/* outputs */

class output {
	friend input;
	friend jive::node;
	friend jive::region;

	typedef std::unordered_set<jive::input*>::const_iterator user_iterator;
public:
	virtual
	~output() noexcept;

	output(
		jive::region * region,
		const jive::port & port);

	output(const output &) = delete;

	output(output &&) = delete;

	output &
	operator=(const output &) = delete;

	output &
	operator=(output &&) = delete;

	inline size_t
	index() const noexcept
	{
		return index_;
	}

	inline size_t
	nusers() const noexcept
	{
		return users_.size();
	}

	inline void
	divert_users(jive::output * new_origin)
	{
		if (this == new_origin)
			return;

		while (users_.size())
			(*users_.begin())->divert_to(new_origin);
	}

	inline user_iterator
	begin() const noexcept
	{
		return users_.begin();
	}

	inline user_iterator
	end() const noexcept
	{
		return users_.end();
	}

	inline const jive::type &
	type() const noexcept
	{
		return port_->type();
	}

	inline jive::region *
	region() const noexcept
	{
		return region_;
	}

	inline const jive::port &
	port() const noexcept
	{
		return *port_;
	}

	virtual std::string
	debug_string() const;

	inline void
	replace(const jive::port & port)
	{
		if (port_->type() != port.type())
			throw type_error(port_->type().debug_string(), port.type().debug_string());

		port_ = port.copy();
	}

  template <class T>
  class iterator
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T*;
    using difference_type = std::ptrdiff_t;
    using pointer = T**;
    using rerefence = T*&;

		static_assert(std::is_base_of<jive::output, T>::value,
			"Template parameter T must be derived from jive::output.");

	protected:
		constexpr
		iterator(T * value)
		: value_(value)
		{}

		virtual T *
		next() const
		{
			/*
				I cannot make this method abstract due to the return value of operator++(int).
				This is the best I could come up with as a workaround.
			*/
			throw compiler_error("This method must be overloaded.");
		}

	public:
		T *
		value() const noexcept
		{
			return value_;
		}

		T &
		operator*()
		{
			JIVE_DEBUG_ASSERT(value_ != nullptr);
			return *value_;
		}

		T *
		operator->() const
		{
			return value_;
		}

		iterator<T> &
		operator++()
		{
			value_ = next();
			return *this;
		}

		iterator<T>
		operator++(int)
		{
			iterator<T> tmp = *this;
			++*this;
			return tmp;
		}

		virtual bool
		operator==(const iterator<T> & other) const
		{
			return value_ == other.value_;
		}

		bool
		operator!=(const iterator<T> & other) const
		{
			return !operator==(other);
		}

	private:
		T * value_;
	};

	template <class T>
  class constiterator
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const T*;
    using difference_type = std::ptrdiff_t;
    using pointer = const T**;
    using reference = const T*&;

		static_assert(std::is_base_of<jive::output, T>::value,
			"Template parameter T must be derived from jive::output.");

	protected:
		constexpr
		constiterator(const T * value)
		: value_(value)
		{}

		virtual const T *
		next() const
		{
			/*
				I cannot make this method abstract due to the return value of operator++(int).
				This is the best I could come up with as a workaround.
			*/
			throw compiler_error("This method must be overloaded.");
		}

	public:
		const T *
		value() const noexcept
		{
			return value_;
		}

		const T &
		operator*()
		{
			JIVE_DEBUG_ASSERT(value_ != nullptr);
			return *value_;
		}

		const T *
		operator->() const
		{
			return value_;
		}

		constiterator<T> &
		operator++()
		{
			value_ = next();
			return *this;
		}

		constiterator<T>
		operator++(int)
		{
			constiterator<T> tmp = *this;
			++*this;
			return tmp;
		}

		virtual bool
		operator==(const constiterator<T> & other) const
		{
			return value_ == other.value_;
		}

		bool
		operator!=(const constiterator<T> & other) const
		{
			return !operator==(other);
		}

	private:
		const T * value_;
	};

private:
	void
	remove_user(jive::input * user);

	void
	add_user(jive::input * user);

	size_t index_;
	jive::region * region_;
	std::unique_ptr<jive::port> port_;
	std::unordered_set<jive::input*> users_;
};

template <class T> static inline bool
is(const jive::output * output) noexcept
{
	static_assert(std::is_base_of<jive::output, T>::value,
		"Template parameter T must be derived from jive::output.");

	return dynamic_cast<const T*>(output) != nullptr;
}

/* node_input class */

class node_input : public jive::input {
public:
	node_input(
		jive::output * origin,
		jive::node * node,
		const jive::port & port);

	jive::node *
	node() const noexcept
	{
		return node_;
	}

  /**
   * Returns the associated node if \p input is a jive::node_input, otherwise null.
   *
   * @param input A jive::input
   * @return Returns a jive::node or null.
   *
   * @see jive::node_input::node()
   */
  [[nodiscard]] static jive::node *
  node(const jive::input & input)
  {
    auto nodeInput = dynamic_cast<const node_input*>(&input);
    return nodeInput != nullptr
           ? nodeInput->node()
           : nullptr;
  }

private:
	jive::node * node_;
};

/* node_output class */

class node_output : public jive::output {
public:
	node_output(
		jive::node * node,
		const jive::port & port);

	jive::node *
	node() const noexcept
	{
		return node_;
	}

	static jive::node *
	node(const jive::output * output)
	{
		auto no = dynamic_cast<const node_output*>(output);
		return no != nullptr ? no->node() : nullptr;
	}

private:
	jive::node * node_;
};

/* node class */

class node {
public:
	virtual
	~node();

	node(std::unique_ptr<jive::operation> op, jive::region * region);

	inline const jive::operation &
	operation() const noexcept
	{
		return *operation_;
	}

	inline bool
	has_users() const noexcept
	{
		for (const auto & output : outputs_) {
			if (output->nusers() != 0)
				return true;
		}

		return false;
	}

	inline bool
	has_predecessors() const noexcept
	{
		for (const auto & input : inputs_) {
			if (is<node_output>(input->origin()))
				return true;
		}

		return false;
	}

	inline bool
	has_successors() const noexcept
	{
		for (const auto & output : outputs_) {
			for (const auto & user : *output) {
				if (is<node_input>(*user))
					return true;
			}
		}

		return false;
	}

	inline size_t
	ninputs() const noexcept
	{
		return inputs_.size();
	}

	node_input *
	input(size_t index) const noexcept
	{
		JIVE_DEBUG_ASSERT(index < ninputs());
		return inputs_[index].get();
	}

	inline size_t
	noutputs() const noexcept
	{
		return outputs_.size();
	}

	node_output *
	output(size_t index) const noexcept
	{
		JIVE_DEBUG_ASSERT(index < noutputs());
		return outputs_[index].get();
	}

	inline void
	recompute_depth() noexcept;

protected:
	node_input *
	add_input(std::unique_ptr<node_input> input);

	void
	remove_input(size_t index);

	node_output *
	add_output(std::unique_ptr<node_output> output)
	{
		output->index_ = noutputs();
		outputs_.push_back(std::move(output));
		return this->output(noutputs()-1);
	}

	void
	remove_output(size_t index);

public:
	inline jive::graph *
	graph() const noexcept
	{
		return graph_;
	}

	inline jive::region *
	region() const noexcept
	{
		return region_;
	}

	virtual jive::node *
	copy(jive::region * region, const std::vector<jive::output*> & operands) const;

	/**
		\brief Copy a node with substitutions
		\param region Target region to create node in
		\param smap Operand substitutions
		\return Copied node

		Create a new node that is semantically equivalent to an
		existing node. The newly created node will use the same
		operands as the existing node unless there is a substitution
		registered for a particular operand.

		The given substitution map is updated so that all
		outputs of the original node will be substituted by
		corresponding outputs of the newly created node in
		subsequent \ref copy operations.
	*/
	virtual jive::node *
	copy(jive::region * region, jive::substitution_map & smap) const = 0;

	inline size_t
	depth() const noexcept
	{
		return depth_;
	}

private:
	jive::detail::intrusive_list_anchor<
		jive::node
	> region_node_list_anchor_;

	jive::detail::intrusive_list_anchor<
		jive::node
	> region_top_node_list_anchor_;

	jive::detail::intrusive_list_anchor<
		jive::node
	> region_bottom_node_list_anchor_;

public:
	typedef jive::detail::intrusive_list_accessor<
		jive::node,
		&jive::node::region_node_list_anchor_
	> region_node_list_accessor;

	typedef jive::detail::intrusive_list_accessor<
		jive::node,
		&jive::node::region_top_node_list_anchor_
	> region_top_node_list_accessor;

	typedef jive::detail::intrusive_list_accessor<
		jive::node,
		&jive::node::region_bottom_node_list_anchor_
	> region_bottom_node_list_accessor;

private:
	size_t depth_;
	jive::graph * graph_;
	jive::region * region_;
	std::unique_ptr<jive::operation> operation_;
	std::vector<std::unique_ptr<node_input>> inputs_;
	std::vector<std::unique_ptr<node_output>> outputs_;
};

static inline std::vector<jive::output*>
operands(const jive::node * node)
{
	std::vector<jive::output*> operands;
	for (size_t n = 0; n < node->ninputs(); n++)
		operands.push_back(node->input(n)->origin());
	return operands;
}

static inline std::vector<jive::output*>
outputs(const jive::node * node)
{
	std::vector<jive::output*> outputs;
	for (size_t n = 0; n < node->noutputs(); n++)
		outputs.push_back(node->output(n));
	return outputs;
}

static inline void
divert_users(
	jive::node * node,
	const std::vector<jive::output*> & outputs)
{
	JIVE_DEBUG_ASSERT(node->noutputs() == outputs.size());

	for (size_t n = 0; n < outputs.size(); n++)
		node->output(n)->divert_users(outputs[n]);
}

template <class T> static inline bool
is(const jive::node * node) noexcept
{
	if (!node)
		return false;

	return is<T>(node->operation());
}

jive::node *
producer(const jive::output * output) noexcept;

bool
normalize(jive::node * node);

}

#endif

/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_DELTA_HPP
#define JLM_IR_OPERATORS_DELTA_HPP

#include <jive/rvsdg/region.hpp>
#include <jive/rvsdg/structural-node.hpp>

#include <jlm/ir/types.hpp>
#include <jlm/ir/variable.hpp>

namespace jlm {

/* delta operator */

class delta_op final : public jive::structural_op {
public:
	inline
	delta_op(
		const ptrtype & type,
		const std::string & name,
		const jlm::linkage & linkage,
		bool constant)
	: constant_(constant)
	, name_(name)
	, linkage_(linkage)
	, type_(type.copy())
	{}

	delta_op(const delta_op & other)
	: constant_(other.constant_)
	, name_(other.name_)
	, linkage_(other.linkage_)
	, type_(other.type_->copy())
	{}

	delta_op(delta_op && other)
	: constant_(other.constant_)
	, name_(std::move(other.name_))
	, linkage_(other.linkage_)
	, type_(std::move(other.type_))
	{}

	delta_op &
	operator=(const delta_op&) = delete;

	delta_op &
	operator=(delta_op&&) = delete;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	virtual bool
	operator==(const operation & other) const noexcept override;

	const std::string &
	name() const noexcept
	{
		return name_;
	}

	const jlm::linkage &
	linkage() const noexcept
	{
		return linkage_;
	}

	inline bool
	constant() const noexcept
	{
		return constant_;
	}

	const ptrtype &
	type() const noexcept
	{
		JLM_DEBUG_ASSERT(dynamic_cast<const ptrtype*>(type_.get()));
		return *static_cast<const ptrtype*>(type_.get());
	}

private:
	bool constant_;
	std::string name_;
	jlm::linkage linkage_;
	std::unique_ptr<jive::type> type_;
};

/* delta node */

class delta_builder;

class delta_node final : public jive::structural_node {
	friend delta_builder;
public:
	virtual
	~delta_node();

private:
	inline
	delta_node(
		jive::region * parent,
		const delta_op & op)
	: jive::structural_node(op, parent, 1)
	{}

	static delta_node *
	create(
		jive::region * parent,
		const ptrtype & type,
		const std::string & name,
		const jlm::linkage & linkage,
		bool constant)
	{
		delta_op op(type, name, linkage, constant);
		return new delta_node(parent, op);
	}

	class iterator final : public std::iterator<std::forward_iterator_tag,
		jive::structural_input*, ptrdiff_t> {

		friend class delta_node;

		constexpr
		iterator(jive::structural_input * input) noexcept
		: input_(input)
		{}

	public:
		jive::structural_input *
		input() const
		{
			return operator->();
		}

		jive::argument *
		argument() const
		{
			return input_->arguments.first();
		}

		const iterator &
		operator++() noexcept
		{
			auto node = input_->node();
			auto index = input_->index();
			input_ = (index == node->ninputs()-1) ? nullptr : node->input(index+1);
			return *this;
		}

		const iterator
		operator++(int) noexcept
		{
			iterator it(*this);
			++(*this);
			return it;
		}

		bool
		operator==(const iterator & other) const noexcept
		{
			return input_ == other.input_;
		}

		bool
		operator!=(const iterator & other) const noexcept
		{
			return !(*this == other);
		}

		jive::structural_input *
		operator->() const noexcept
		{
			return input_;
		}

		jive::structural_input &
		operator*() const noexcept
		{
			JLM_DEBUG_ASSERT(input_ != nullptr);
			return *input_;
		}

	private:
		jive::structural_input * input_;
	};

public:
	inline jive::region *
	subregion() const noexcept
	{
		return jive::structural_node::subregion(0);
	}

	delta_node::iterator
	begin() const
	{
		if (ninputs() == 0)
			return end();

		return iterator(input(0));
	}

	delta_node::iterator
	end() const
	{
		return iterator(nullptr);
	}

	inline jive::argument *
	add_dependency(jive::output * origin)
	{
		auto input = jive::structural_input::create(this, origin, origin->type());
		return jive::argument::create(subregion(), input, input->port());
	}

	const std::string &
	name() const noexcept
	{
		return static_cast<const delta_op*>(&operation())->name();
	}

	const jlm::linkage &
	linkage() const noexcept
	{
		return static_cast<const delta_op*>(&operation())->linkage();
	}

	bool
	constant() const noexcept
	{
		return static_cast<const delta_op*>(&operation())->constant();
	}

	const ptrtype &
	type() const noexcept
	{
		return static_cast<const delta_op*>(&operation())->type();
	}

	virtual delta_node *
	copy(jive::region * region, jive::substitution_map & smap) const override;
};

/* delta builder */

class delta_builder final {
public:
	inline constexpr
	delta_builder() noexcept
	: node_(nullptr)
	{}

	delta_builder(const delta_builder &) = delete;

	delta_builder(delta_builder &&) = delete;

	delta_builder &
	operator=(const delta_builder &) = delete;

	delta_builder &
	operator=(delta_builder &&) = delete;

	inline jive::region *
	region() const noexcept
	{
		return node_ ? node_->subregion() : nullptr;
	}

	inline jive::region *
	begin(
		jive::region * parent,
		const ptrtype & type,
		const std::string & name,
		const jlm::linkage & linkage,
		bool constant)
	{
		if (node_)
			return region();

		node_ = delta_node::create(parent, type, name, linkage, constant);
		return region();
	}

	inline jive::output *
	add_dependency(jive::output * origin)
	{
		return node_ ? node_->add_dependency(origin) : nullptr;
	}

	inline jive::output *
	end(jive::output * data)
	{
		if (!node_)
			return nullptr;

		if (data->type() != node_->type().pointee_type())
			throw jlm::error("Invalid type.");

		auto output = jive::structural_output::create(node_, node_->type());
		jive::result::create(region(), data, output, data->type());
		node_ = nullptr;

		return output;
	}

private:
	delta_node * node_;
};

}

#endif

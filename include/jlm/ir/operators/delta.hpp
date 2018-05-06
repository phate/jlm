/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_DELTA_HPP
#define JLM_IR_OPERATORS_DELTA_HPP

#include <jive/rvsdg/region.h>
#include <jive/rvsdg/structural-node.h>

#include <jlm/ir/types.hpp>
#include <jlm/ir/variable.hpp>

namespace jlm {

/* delta operator */

class delta_op final : public jive::structural_op {
public:
	inline constexpr
	delta_op(const jlm::linkage & linkage, bool constant)
	: constant_(constant)
	, linkage_(linkage)
	{}

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	virtual bool
	operator==(const operation & other) const noexcept override;

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

private:
	bool constant_;
	jlm::linkage linkage_;
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
		const jlm::linkage & linkage,
		bool constant)
	{
		delta_op op(linkage, constant);
		return new delta_node(parent, op);
	}

	class iterator final {
	public:
		inline constexpr
		iterator(jive::input * input) noexcept
		: input_(input)
		{}

		inline const iterator &
		operator++() noexcept
		{
			auto node = input_->node();
			auto index = input_->index();
			input_ = (index == node->ninputs()-1) ? nullptr : node->input(index+1);
			return *this;
		}

		inline const iterator
		operator++(int) noexcept
		{
			iterator it(*this);
			++(*this);
			return it;
		}

		inline bool
		operator==(const iterator & other) const noexcept
		{
			return input_ == other.input_;
		}

		inline bool
		operator!=(const iterator & other) const noexcept
		{
			return !(*this == other);
		}

		inline jive::input *
		operator*() noexcept
		{
			return input_;
		}

	private:
		jive::input * input_;
	};

public:
	inline jive::region *
	subregion() const noexcept
	{
		return jive::structural_node::subregion(0);
	}

	inline delta_node::iterator
	begin() const
	{
		return iterator(subregion()->argument(0)->input());
	}

	inline delta_node::iterator
	end() const
	{
		return iterator(nullptr);
	}

	inline jive::argument *
	add_dependency(jive::output * origin)
	{
		auto input = add_input(origin->type(), origin);
		return subregion()->add_argument(input, origin->type());
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
		const jlm::linkage & linkage,
		bool constant)
	{
		if (node_)
			return region();

		node_ = delta_node::create(parent, linkage, constant);
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

		auto output = node_->add_output({std::move(create_ptrtype(data->type()))});
		region()->add_result(data, output, data->type());
		return output;
	}

private:
	delta_node * node_;
};

}

#endif

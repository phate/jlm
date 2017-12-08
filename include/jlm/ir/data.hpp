/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_DATA_HPP
#define JLM_IR_DATA_HPP

#include <jive/rvsdg/region.h>
#include <jive/rvsdg/structural-node.h>

#include <jlm/ir/types.hpp>
#include <jlm/ir/variable.hpp>

namespace jlm {

/* data operator */

class data_op final : public jive::structural_op {
public:
	inline constexpr
	data_op(const jlm::linkage & linkage, bool constant)
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

static inline bool
is_data_op(const jive::operation & op) noexcept
{
	return dynamic_cast<const jlm::data_op*>(&op) != nullptr;
}

/* data builder */

class data_builder final {
public:
	inline constexpr
	data_builder() noexcept
	: node_(nullptr)
	{}

	data_builder(const data_builder &) = delete;

	data_builder(data_builder &&) = delete;

	data_builder &
	operator=(const data_builder &) = delete;

	data_builder &
	operator=(data_builder &&) = delete;

	inline jive::region *
	region() const noexcept
	{
		return node_ ? node_->subregion(0) : nullptr;
	}

	inline jive::region *
	begin(
		jive::region * parent,
		const jlm::linkage & linkage,
		bool constant)
	{
		if (node_)
			return region();

		node_ = parent->add_structural_node(jlm::data_op(linkage, constant), 1);
		return region();
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
	jive::structural_node * node_;
};

}

#endif

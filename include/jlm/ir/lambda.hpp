/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_LAMBDA_HPP
#define JLM_IR_LAMBDA_HPP

#include <jive/rvsdg/region.h>
#include <jive/rvsdg/structural-node.h>
#include <jive/types/function/fcttype.h>

#include <jlm/ir/types.hpp>

namespace jlm {

/* lambda operation */

class lambda_op final : public jive::structural_op {
public:
	virtual
	~lambda_op();

	inline
	lambda_op(jive::fct::type fcttype) noexcept
	: fcttype_(fcttype)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	inline const jive::fct::type &
	fcttype() const noexcept
	{
		return fcttype_;
	}

	virtual std::unique_ptr<jive::operation>
	copy() const override;

private:
	jive::fct::type fcttype_;
};

static inline bool
is_lambda_op(const jive::operation & op) noexcept
{
	return dynamic_cast<const lambda_op*>(&op) != nullptr;
}

static inline bool
is_lambda_node(const jive::node * node) noexcept
{
	return jive::is_opnode<lambda_op>(node);
}

/* lambda node */

class lambda_builder;

class lambda_node final : public jive::structural_node {
	friend lambda_builder;
public:
	virtual
	~lambda_node();

private:
	inline
	lambda_node(jive::region * parent, jive::fct::type fcttype)
	: jive::structural_node(lambda_op(std::move(fcttype)), parent, 1)
	{}

	static lambda_node *
	create(jive::region * parent, jive::fct::type fcttype)
	{
		return new lambda_node(parent, std::move(fcttype));
	}

	class dependency_iterator final {
	public:
		inline constexpr
		dependency_iterator(jive::input * input) noexcept
		: input_(input)
		{}

		inline const dependency_iterator &
		operator++() noexcept
		{
			auto node = input_->node();
			auto index = input_->index();
			input_ = (index == node->ninputs()-1) ? nullptr : node->input(index+1);
			return *this;
		}

		inline const dependency_iterator
		operator++(int) noexcept
		{
			dependency_iterator it(*this);
			++(*this);
			return it;
		}

		inline bool
		operator==(const dependency_iterator & other) const noexcept
		{
			return input_ == other.input_;
		}

		inline bool
		operator!=(const dependency_iterator & other) const noexcept
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

	inline lambda_node::dependency_iterator
	begin() const
	{
		auto argument = subregion()->argument(0);
		while (argument && argument->input() == nullptr)
			argument = subregion()->argument(argument->index()+1);

		return dependency_iterator(argument->input());
	}

	inline lambda_node::dependency_iterator
	end() const
	{
		return dependency_iterator(nullptr);
	}

	inline std::vector<jive::argument*>
	arguments() const noexcept
	{
		std::vector<jive::argument*> arguments;

		auto argument = subregion()->argument(0);
		while (argument && argument->input() == nullptr) {
			arguments.push_back(argument);
			argument = subregion()->argument(argument->index()+1);
		}

		return arguments;
	}

	inline jive::argument *
	add_dependency(jive::output * origin)
	{
		auto input = add_input(origin->type(), origin);
		return subregion()->add_argument(input, origin->type());
	}

	inline const jive::fct::type &
	fcttype() const noexcept
	{
		return static_cast<const lambda_op*>(&operation())->fcttype();
	}

	virtual lambda_node *
	copy(jive::region * region, jive::substitution_map & smap) const override;
};

/* lambda builder */

class lambda_builder final {
public:
	inline
	lambda_builder()
	: lambda_(nullptr)
	{}

	inline std::vector<jive::argument*>
	begin_lambda(jive::region * parent, jive::fct::type type)
	{
		if (lambda_)
			return lambda_->arguments();

		std::vector<jive::argument*> arguments;
		lambda_ = lambda_node::create(parent, std::move(type));
		for (size_t n = 0; n < lambda_->fcttype().narguments(); n++) {
			auto & type = lambda_->fcttype().argument_type(n);
			arguments.push_back(lambda_->subregion()->add_argument(nullptr, type));
		}

		return arguments;
	}

	inline jive::region *
	subregion() const noexcept
	{
		return lambda_ ? lambda_->subregion() : nullptr;
	}

	inline jive::output *
	add_dependency(jive::output * origin)
	{
		return lambda_ ? lambda_->add_dependency(origin) : nullptr;
	}

	inline lambda_node *
	end_lambda(const std::vector<jive::output*> & results)
	{
		if (!lambda_)
			return nullptr;

		const auto & fcttype = lambda_->fcttype();
		if (results.size() != fcttype.nresults())
			throw std::logic_error("Incorrect number of results.");

		for (size_t n = 0; n < results.size(); n++)
			lambda_->subregion()->add_result(results[n], nullptr, fcttype.result_type(n));
		lambda_->add_output(ptrtype(fcttype));

		auto lambda = lambda_;
		lambda_ = nullptr;
		return lambda;
	}

private:
	lambda_node * lambda_;
};

}

#endif

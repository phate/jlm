/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_LAMBDA_HPP
#define JLM_IR_OPERATORS_LAMBDA_HPP

#include <jive/rvsdg/region.h>
#include <jive/rvsdg/structural-node.h>
#include <jive/types/function.h>

#include <jlm/ir/types.hpp>
#include <jlm/ir/variable.hpp>

namespace jlm {

/* lambda operation */

class lambda_op final : public jive::lambda_op {
public:
	virtual
	~lambda_op();

	inline
	lambda_op(
		jive::fcttype fcttype,
		const std::string & name,
		const jlm::linkage & linkage) noexcept
	: jive::lambda_op(std::move(fcttype))
	, name_(name)
	, linkage_(linkage)
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	inline const jive::fcttype &
	fcttype() const noexcept
	{
		return function_type();
	}

	inline const std::string &
	name() const noexcept
	{
		return name_;
	}

	inline const jlm::linkage &
	linkage() const noexcept
	{
		return linkage_;
	}

	virtual std::unique_ptr<jive::operation>
	copy() const override;

private:
	std::string name_;
	jlm::linkage linkage_;
};

/* lambda node */

class lambda_builder;

class lambda_node final : public jive::structural_node {
	friend lambda_builder;
public:
	virtual
	~lambda_node();

private:
	inline
	lambda_node(jive::region * parent, const jlm::lambda_op & op)
	: jive::structural_node(op, parent, 1)
	{}

	static lambda_node *
	create(jive::region * parent, const jlm::lambda_op & op)
	{
		return new lambda_node(parent, op);
	}

	class argument_iterator final : public std::iterator<std::forward_iterator_tag,
		jive::argument*, ptrdiff_t> {

		friend class lambda_node;

		using super = std::iterator<std::forward_iterator_tag, jive::argument*, ptrdiff_t>;

		argument_iterator(jive::argument * argument)
		: argument_(argument)
		{}

	public:
		jive::argument *
		argument() const
		{
			return operator->();
		}

		jive::argument &
		operator*() const
		{
			JLM_DEBUG_ASSERT(argument_ != nullptr);
			return *argument_;
		}

		jive::argument *
		operator->() const
		{
			return &operator*();
		}

		argument_iterator &
		operator++()
		{
			if (argument_ == nullptr) {
				argument_ = nullptr;
				return *this;
			}

			auto region = argument_->region();
			auto index = argument_->index() + 1;
			if (index >= region->narguments()) {
				argument_ = nullptr;
				return *this;
			}

			auto argument = region->argument(index);
			argument_ = argument->input() != nullptr ? nullptr : argument;

			return *this;
		}

		argument_iterator
		operator++(int)
		{
			argument_iterator tmp = *this;
			++*this;
			return tmp;
		}

		bool
		operator==(const argument_iterator & other) const
		{
			return argument_ == other.argument_;
		}

		bool
		operator!=(const argument_iterator & other) const
		{
			return !operator==(other);
		}

	private:
		jive::argument * argument_;
	};

	class cv_iterator final : public std::iterator<std::forward_iterator_tag,
		jive::structural_input*, ptrdiff_t> {

		friend class lambda_node;

		constexpr
		cv_iterator(jive::structural_input * input) noexcept
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

		jive::structural_input &
		operator*() const
		{
			JLM_DEBUG_ASSERT(input_ != nullptr);
			return *input_;
		}

		jive::structural_input *
		operator->() const
		{
			return &operator*();
		}

		const cv_iterator &
		operator++() noexcept
		{
			auto node = input_->node();
			auto index = input_->index();
			input_ = (index == node->ninputs()-1) ? nullptr : node->input(index+1);
			return *this;
		}

		const cv_iterator
		operator++(int) noexcept
		{
			cv_iterator it(*this);
			++(*this);
			return it;
		}

		bool
		operator==(const cv_iterator & other) const noexcept
		{
			return input_ == other.input_;
		}

		bool
		operator!=(const cv_iterator & other) const noexcept
		{
			return !(*this == other);
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

	argument_iterator
	begin_argument() const
	{
		if (subregion()->narguments() == 0
		|| subregion()->argument(0)->input() != nullptr)
			return end_argument();

		return argument_iterator(subregion()->argument(0));
	}

	argument_iterator
	end_argument() const
	{
		return argument_iterator(nullptr);
	}

	cv_iterator
	begin_cv() const
	{
		if (ninputs() == 0)
			return end_cv();

		return cv_iterator(input(0));
	}

	cv_iterator
	end_cv() const
	{
		return cv_iterator(nullptr);
	}

	cv_iterator
	begin() const
	{
		return begin_cv();
	}

	cv_iterator
	end() const
	{
		return end_cv();
	}

	std::vector<jive::argument*>
	arguments() const noexcept
	{
		std::vector<jive::argument*> arguments;
		for (auto it = begin_argument(); it != end_argument(); it++)
			arguments.push_back(it.argument());

		return arguments;
	}

	size_t
	narguments() const noexcept
	{
		JLM_DEBUG_ASSERT(subregion()->narguments() >= ninputs());
		return subregion()->narguments() - ninputs();
	}

	inline jive::argument *
	add_dependency(jive::output * origin)
	{
		auto input = add_input(origin->type(), origin);
		return subregion()->add_argument(input, origin->type());
	}

	inline const jive::fcttype &
	fcttype() const noexcept
	{
		return static_cast<const lambda_op*>(&operation())->fcttype();
	}

	inline const std::string &
	name() const noexcept
	{
		return static_cast<const lambda_op*>(&operation())->name();
	}

	inline const jlm::linkage &
	linkage() const noexcept
	{
		return static_cast<const lambda_op*>(&operation())->linkage();
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
	begin_lambda(jive::region * parent, const jlm::lambda_op & op)
	{
		if (lambda_)
			return lambda_->arguments();

		std::vector<jive::argument*> arguments;
		lambda_ = lambda_node::create(parent, op);
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
			throw jlm::error("incorrect number of results.");

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

static inline bool
is_lambda_argument(const jive::output * output)
{
	auto argument = dynamic_cast<const jive::argument*>(output);
	return argument && jive::is<lambda_op>(argument->region()->node());
}

static inline bool
is_lambda_output(const jive::output * output)
{
	return is<lambda_op>(output->node());
}

static inline bool
is_lambda_cv(const jive::output * output)
{
	auto argument = dynamic_cast<const jive::argument*>(output);
	return argument
	    && is<lambda_op>(argument->region()->node())
	    && argument->input() != nullptr;
}

}

#endif

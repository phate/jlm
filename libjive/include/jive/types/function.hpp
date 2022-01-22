/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JIVE_TYPES_FUNCTION_HPP
#define JIVE_TYPES_FUNCTION_HPP

#include <jive/rvsdg/simple-node.hpp>
#include <jive/rvsdg/structural-node.hpp>
#include <jive/rvsdg/type.hpp>

namespace jive {

/* function type */

class fcttype final : public jive::valuetype {
public:
	virtual
	~fcttype() noexcept;

	fcttype(
		const std::vector<const jive::type*> & argument_types,
		const std::vector<const jive::type*> & result_types);

	fcttype(
		const std::vector<std::unique_ptr<jive::type>> & argument_types,
		const std::vector<std::unique_ptr<jive::type>> & result_types);

	fcttype(const fcttype & other);

	fcttype(fcttype && other);

	fcttype &
	operator=(const fcttype & other);

	fcttype &
	operator=(fcttype && other);

	inline size_t
	nresults() const noexcept
	{
		return result_types_.size();
	}

	inline size_t
	narguments() const noexcept
	{
		return argument_types_.size();
	}

	inline const jive::type &
	result_type(size_t index) const noexcept
	{
		return *result_types_[index];
	}

	inline const jive::type &
	argument_type(size_t index) const noexcept
	{
		return *argument_types_[index];
	}

	virtual std::string
	debug_string() const override;

	virtual bool
	operator==(const jive::type & other) const noexcept override;

	virtual std::unique_ptr<jive::type>
	copy() const override;

private:
	std::vector<std::unique_ptr<jive::type>> result_types_;
	std::vector<std::unique_ptr<jive::type>> argument_types_;
};

/* apply operator */

class apply_op final : public jive::simple_op {
public:
	virtual
	~apply_op() noexcept;

	inline
	apply_op(const fcttype & type)
	: simple_op(create_operands(type), create_results(type))
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	inline const fcttype &
	function_type() const noexcept
	{
		return *static_cast<const jive::fcttype*>(&argument(0).type());
	}

	virtual std::unique_ptr<jive::operation>
	copy() const override;

private:
	static std::vector<jive::port>
	create_operands(const fcttype & type);

	static std::vector<jive::port>
	create_results(const fcttype & type);
};

static inline std::vector<jive::output*>
create_apply(jive::output * function, const std::vector<jive::output*> & arguments)
{
	auto ft = dynamic_cast<const fcttype*>(&function->type());
	if (!ft) throw type_error("fct", function->type().debug_string());

	apply_op op(*ft);
	std::vector<jive::output*> operands({function});
	operands.insert(operands.end(), arguments.begin(), arguments.end());

	return simple_node::create_normalized(function->region(), op, operands);
}

/* lambda operator */

class lambda_op : public structural_op {
public:
	virtual
	~lambda_op() noexcept;

	inline
	lambda_op(const lambda_op & other) = default;

	inline
	lambda_op(lambda_op && other) = default;

	inline
	lambda_op(jive::fcttype function_type) noexcept
		: function_type_(std::move(function_type))
	{}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	inline const jive::fcttype &
	function_type() const noexcept
	{
		return function_type_;
	}

	virtual std::unique_ptr<jive::operation>
	copy() const override;

private:
	fcttype function_type_;
};

/* lambda node */

class argument;
class lambda_builder;

class lambda_node final : public jive::structural_node {
	friend lambda_builder;
public:
	virtual
	~lambda_node();

private:
	inline
	lambda_node(jive::region * parent, fcttype type)
	: jive::structural_node(jive::lambda_op(std::move(type)), parent, 1)
	{}

	inline
	lambda_node(jive::region * parent, const lambda_op & op)
	: structural_node(op, parent, 1)
	{}

	class dependency_iterator {
	public:
		inline constexpr
		dependency_iterator(structural_input * input) noexcept
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
		structural_input * input_;
	};

	static jive::lambda_node *
	create(jive::region * parent, fcttype type)
	{
		return new jive::lambda_node(parent, std::move(type));
	}

	static jive::lambda_node *
	create(jive::region * parent, const jive::lambda_op & op)
	{
		return new jive::lambda_node(parent, op);
	}

public:
	inline jive::region *
	subregion() const noexcept
	{
		return structural_node::subregion(0);
	}

	inline lambda_node::dependency_iterator
	begin() const
	{
		auto argument = subregion()->argument(0);
		while (argument->input() == nullptr && argument != nullptr)
			argument = subregion()->argument(argument->index()+1);

		return dependency_iterator(argument->input());
	}

	inline lambda_node::dependency_iterator
	end() const
	{
		return dependency_iterator(nullptr);
	}

	inline jive::argument *
	add_dependency(jive::output * origin)
	{
		auto input = structural_input::create(this, origin, origin->type());
		return argument::create(subregion(), input, origin->type());
	}

	inline const fcttype &
	function_type() const noexcept
	{
		return static_cast<const lambda_op*>(&operation())->function_type();
	}

	virtual jive::lambda_node *
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
	begin_lambda(jive::region * parent, const lambda_op & op)
	{
		std::vector<jive::argument*> arguments;

		if (lambda_) {
			auto argument = lambda_->subregion()->argument(0);
			while (argument->input() == nullptr && argument != nullptr) {
				arguments.push_back(argument);
				argument = lambda_->subregion()->argument(argument->index()+1);
			}
			return arguments;
		}

		lambda_ = jive::lambda_node::create(parent, op);
		for (size_t n = 0; n < lambda_->function_type().narguments(); n++) {
			auto & argument_type = lambda_->function_type().argument_type(n);
			arguments.push_back(argument::create(lambda_->subregion(), nullptr, argument_type));
		}
		return arguments;
	}

	inline std::vector<jive::argument*>
	begin_lambda(jive::region * parent, fcttype type)
	{
		return begin_lambda(parent, lambda_op(std::move(type)));
	}

	inline jive::region *
	subregion() const noexcept
	{
		return lambda_ ? lambda_->subregion() : nullptr;
	}

	inline jive::output *
	add_dependency(jive::output * value)
	{
		return lambda_ ? lambda_->add_dependency(value) : nullptr;
	}

	inline jive::lambda_node *
	end_lambda(const std::vector<jive::output*> & results)
	{
		if (!lambda_)
			return nullptr;

		const auto & ftype = lambda_->function_type();
		if (results.size() != ftype.nresults())
			throw jive::compiler_error("Incorrect number of results.");

		for (size_t n = 0; n < results.size(); n++)
			result::create(lambda_->subregion(), results[n], nullptr, ftype.result_type(n));
		structural_output::create(lambda_, ftype);

		auto lambda = lambda_;
		lambda_ = nullptr;
		return lambda;
	}

private:
	jive::lambda_node * lambda_;
};

}

#endif

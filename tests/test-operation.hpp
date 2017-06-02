/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef TESTS_TEST_OPERATION_HPP
#define TESTS_TEST_OPERATION_HPP

#include <jive/vsdg/basetype.h>
#include <jive/vsdg/operators/operation.h>

#include <jlm/IR/tac.hpp>

namespace jlm {

class test_op final : public jive::operation {
public:
	virtual
	~test_op() noexcept;

	inline
	test_op(
		const std::vector<const jive::base::type*> & argument_types,
		const std::vector<const jive::base::type*> & result_types)
	: operation()
	{
		for (const auto & type : argument_types)
			argument_types_.emplace_back(type->copy());
		for (const auto & type : result_types)
			result_types_.emplace_back(type->copy());
	}

	inline
	test_op(const test_op & other)
	: operation()
	{
		for (const auto & type : other.argument_types_)
			argument_types_.emplace_back(type->copy());
		for (const auto & type : other.result_types_)
			result_types_.emplace_back(type->copy());
	}

	inline test_op &
	operator=(const test_op & other)
	{
		if (this == &other)
			return *this;

		argument_types_.clear();
		for (const auto & type : other.argument_types_)
			argument_types_.emplace_back(type->copy());

		result_types_.clear();
		for (const auto & type : other.result_types_)
			result_types_.emplace_back(type->copy());

		return *this;
	}

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual size_t
	narguments() const noexcept override;

	virtual const jive::base::type &
	argument_type(size_t index) const noexcept override;

	virtual size_t
	nresults() const noexcept override;

	virtual const jive::base::type &
	result_type(size_t index) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

private:
	std::vector<std::unique_ptr<jive::base::type>> result_types_;
	std::vector<std::unique_ptr<jive::base::type>> argument_types_;
};

static inline std::unique_ptr<jlm::tac>
create_testop_tac(
	const std::vector<const variable*> & arguments,
	const std::vector<const variable*> & results)
{
	std::vector<const jive::base::type*> result_types;
	std::vector<const jive::base::type*> argument_types;
	for (const auto & arg : arguments)
		argument_types.push_back(&arg->type());
	for (const auto & res : results)
		result_types.push_back(&res->type());

	test_op op(argument_types, result_types);
	return create_tac(op, arguments, results);
}

}

#endif

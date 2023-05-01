/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef TESTS_TEST_OPERATION_HPP
#define TESTS_TEST_OPERATION_HPP

#include <jlm/rvsdg/binary.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/operation.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/type.hpp>
#include <jlm/rvsdg/unary.hpp>

#include <jlm/llvm/ir/tac.hpp>

namespace jlm {

/* unary operation */

class unary_op final : public jive::unary_op {
public:
	virtual
	~unary_op() noexcept;

	inline
	unary_op(
		const jive::port & srcport,
		const jive::port & dstport) noexcept
	: jive::unary_op(srcport, dstport)
	{}

	virtual bool
	operator==(const jive::operation & other) const noexcept override;

	virtual jive_unop_reduction_path_t
	can_reduce_operand(
		const jive::output * operand) const noexcept override;

	virtual jive::output *
	reduce_operand(
		jive_unop_reduction_path_t path,
		jive::output * operand) const override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static inline jive::node *
	create(
		jive::region * region,
		const jive::port & srcport,
		jive::output * operand,
		const jive::port & dstport)
	{
		return jive::simple_node::create(region, unary_op(srcport, dstport), {operand});
	}

	static inline jive::output *
	create_normalized(
		const jive::port & srcport,
		jive::output * operand,
		const jive::port & dstport)
	{
		unary_op op(srcport, dstport);
		return jive::simple_node::create_normalized(operand->region(), op, {operand})[0];
	}
};

static inline bool
is_unary_op(const jive::operation & op) noexcept
{
	return dynamic_cast<const unary_op*>(&op);
}

static inline bool
is_unary_node(const jive::node * node) noexcept
{
	return is<unary_op>(node);
}

/* binary operation */

class binary_op final : public jive::binary_op {
public:
	virtual
	~binary_op() noexcept;

	inline
	binary_op(
		const jive::port & srcport,
		const jive::port & dstport,
		const enum jive::binary_op::flags & flags) noexcept
	: jive::binary_op({srcport, srcport}, {dstport})
	, flags_(flags)
	{}

	virtual bool
	operator==(const jive::operation & other) const noexcept override;

	virtual jive_binop_reduction_path_t
	can_reduce_operand_pair(
		const jive::output * op1,
		const jive::output * op2) const noexcept override;

	virtual jive::output *
	reduce_operand_pair(
		jive_unop_reduction_path_t path,
		jive::output * op1,
		jive::output * op2) const override;

	virtual enum jive::binary_op::flags
	flags() const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static inline jive::node *
	create(
		const jive::port & srcport,
		const jive::port & dstport,
		jive::output * op1,
		jive::output * op2)
	{
		binary_op op(srcport, dstport, jive::binary_op::flags::none);
		return jive::simple_node::create(op1->region(), op, {op1, op2});
	}

	static inline jive::output *
	create_normalized(
		const jive::port & srcport,
		const jive::port & dstport,
		jive::output * op1,
		jive::output * op2)
	{
		binary_op op(srcport, dstport, jive::binary_op::flags::none);
		return jive::simple_node::create_normalized(op1->region(), op, {op1, op2})[0];
	}

private:
	enum jive::binary_op::flags flags_;
};

/* structural operation */

class structural_op final : public jive::structural_op {
public:
	virtual
	~structural_op() noexcept;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;
};

class structural_node final : public jive::structural_node {
public:
	~structural_node() override;

private:
	structural_node(
		jive::region * parent,
		size_t nsubregions)
	: jive::structural_node(structural_op(), parent, nsubregions)
	{}

public:
	static structural_node *
	create(
		jive::region * parent,
		size_t nsubregions)
	{
		return new structural_node(parent, nsubregions);
	}

	virtual structural_node *
	copy(jive::region * region, jive::substitution_map & smap) const override;
};

class test_op final : public jive::simple_op {
public:
	virtual
	~test_op();

	inline
	test_op(
		const std::vector<const jive::type*> & arguments,
		const std::vector<const jive::type*> & results)
	: simple_op(create_ports(arguments), create_ports(results))
	{}

	test_op(const test_op &) = default;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<jive::operation>
	copy() const override;

	static jive::simple_node *
	create(
		jive::region * region,
		const std::vector<jive::output*> & operands,
		const std::vector<const jive::type*> & result_types)
	{
		std::vector<const jive::type*> operand_types;
		for (const auto & operand : operands)
			operand_types.push_back(&operand->type());

		test_op op(operand_types, result_types);
		return jive::simple_node::create(region, op, {operands});
	}

	static jive::simple_node *
	Create(
		jive::region * region,
		const std::vector<const jive::type*> & operandTypes,
		const std::vector<jive::output*> & operands,
		const std::vector<const jive::type*> & resultTypes)
	{
		test_op op(operandTypes, resultTypes);
		return jive::simple_node::create(region, op, {operands});
	}

private:
	static inline std::vector<jive::port>
	create_ports(const std::vector<const jive::type*> & types)
	{
		std::vector<jive::port> ports;
		for (const auto & type : types)
			ports.push_back({*type});

		return ports;
	}
};

static inline std::unique_ptr<jlm::tac>
create_testop_tac(
	const std::vector<const variable*> & arguments,
	const std::vector<const jive::type*> & result_types)
{
	std::vector<const jive::type*> argument_types;
	for (const auto & arg : arguments)
		argument_types.push_back(&arg->type());

	test_op op(argument_types, result_types);
	return tac::create(op, arguments);
}

static inline std::vector<jive::output*>
create_testop(
	jive::region * region,
	const std::vector<jive::output*> & operands,
	const std::vector<const jive::type*> & result_types)
{
	std::vector<const jive::type*> operand_types;
	for (const auto & operand : operands)
		operand_types.push_back(&operand->type());

	test_op op(operand_types, result_types);
	return jive::simple_node::create_normalized(region, op, {operands});
}

}

#endif

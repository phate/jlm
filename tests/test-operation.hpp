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

class unary_op final : public rvsdg::unary_op {
public:
	virtual
	~unary_op() noexcept;

	inline
	unary_op(
		const rvsdg::port & srcport,
		const rvsdg::port & dstport) noexcept
	: rvsdg::unary_op(srcport, dstport)
	{}

	virtual bool
	operator==(const rvsdg::operation & other) const noexcept override;

	virtual rvsdg::unop_reduction_path_t
	can_reduce_operand(
		const rvsdg::output * operand) const noexcept override;

	virtual rvsdg::output *
	reduce_operand(
		rvsdg::unop_reduction_path_t path,
		rvsdg::output * operand) const override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<rvsdg::operation>
	copy() const override;

	static inline rvsdg::node *
	create(
		rvsdg::region * region,
		const rvsdg::port & srcport,
		rvsdg::output * operand,
		const rvsdg::port & dstport)
	{
		return rvsdg::simple_node::create(region, unary_op(srcport, dstport), {operand});
	}

	static inline rvsdg::output *
	create_normalized(
		const rvsdg::port & srcport,
		rvsdg::output * operand,
		const rvsdg::port & dstport)
	{
		unary_op op(srcport, dstport);
		return rvsdg::simple_node::create_normalized(operand->region(), op, {operand})[0];
	}
};

static inline bool
is_unary_op(const rvsdg::operation & op) noexcept
{
	return dynamic_cast<const unary_op*>(&op);
}

static inline bool
is_unary_node(const rvsdg::node * node) noexcept
{
	return jlm::rvsdg::is<unary_op>(node);
}

/* binary operation */

class binary_op final : public rvsdg::binary_op {
public:
	virtual
	~binary_op() noexcept;

	inline
	binary_op(
		const rvsdg::port & srcport,
		const rvsdg::port & dstport,
		const enum rvsdg::binary_op::flags & flags) noexcept
	: rvsdg::binary_op({srcport, srcport}, {dstport})
	, flags_(flags)
	{}

	virtual bool
	operator==(const rvsdg::operation & other) const noexcept override;

	virtual rvsdg::binop_reduction_path_t
	can_reduce_operand_pair(
		const rvsdg::output * op1,
		const rvsdg::output * op2) const noexcept override;

	virtual rvsdg::output *
	reduce_operand_pair(
		rvsdg::unop_reduction_path_t path,
		rvsdg::output * op1,
		rvsdg::output * op2) const override;

	virtual enum rvsdg::binary_op::flags
	flags() const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<rvsdg::operation>
	copy() const override;

	static inline rvsdg::node *
	create(
		const rvsdg::port & srcport,
		const rvsdg::port & dstport,
		rvsdg::output * op1,
		rvsdg::output * op2)
	{
		binary_op op(srcport, dstport, rvsdg::binary_op::flags::none);
		return rvsdg::simple_node::create(op1->region(), op, {op1, op2});
	}

	static inline rvsdg::output *
	create_normalized(
		const rvsdg::port & srcport,
		const rvsdg::port & dstport,
		rvsdg::output * op1,
		rvsdg::output * op2)
	{
		binary_op op(srcport, dstport, rvsdg::binary_op::flags::none);
		return rvsdg::simple_node::create_normalized(op1->region(), op, {op1, op2})[0];
	}

private:
	enum rvsdg::binary_op::flags flags_;
};

/* structural operation */

class structural_op final : public rvsdg::structural_op {
public:
	virtual
	~structural_op() noexcept;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<rvsdg::operation>
	copy() const override;
};

class structural_node final : public rvsdg::structural_node {
public:
	~structural_node() override;

private:
	structural_node(
		rvsdg::region * parent,
		size_t nsubregions)
	: rvsdg::structural_node(structural_op(), parent, nsubregions)
	{}

public:
	static structural_node *
	create(
		rvsdg::region * parent,
		size_t nsubregions)
	{
		return new structural_node(parent, nsubregions);
	}

	virtual structural_node *
	copy(rvsdg::region * region, rvsdg::substitution_map & smap) const override;
};

class test_op final : public rvsdg::simple_op {
public:
	virtual
	~test_op();

	inline
	test_op(
		const std::vector<const rvsdg::type*> & arguments,
		const std::vector<const rvsdg::type*> & results)
	: simple_op(create_ports(arguments), create_ports(results))
	{}

	test_op(const test_op &) = default;

	virtual bool
	operator==(const operation & other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual std::unique_ptr<rvsdg::operation>
	copy() const override;

	static rvsdg::simple_node *
	create(
		rvsdg::region * region,
		const std::vector<rvsdg::output*> & operands,
		const std::vector<const rvsdg::type*> & result_types)
	{
		std::vector<const rvsdg::type*> operand_types;
		for (const auto & operand : operands)
			operand_types.push_back(&operand->type());

		test_op op(operand_types, result_types);
		return rvsdg::simple_node::create(region, op, {operands});
	}

	static rvsdg::simple_node *
	Create(
		rvsdg::region * region,
		const std::vector<const rvsdg::type*> & operandTypes,
		const std::vector<rvsdg::output*> & operands,
		const std::vector<const rvsdg::type*> & resultTypes)
	{
		test_op op(operandTypes, resultTypes);
		return rvsdg::simple_node::create(region, op, {operands});
	}

private:
	static inline std::vector<rvsdg::port>
	create_ports(const std::vector<const rvsdg::type*> & types)
	{
		std::vector<rvsdg::port> ports;
		for (const auto & type : types)
			ports.push_back({*type});

		return ports;
	}
};

static inline std::unique_ptr<llvm::tac>
create_testop_tac(
	const std::vector<const llvm::variable*> & arguments,
	const std::vector<const rvsdg::type*> & result_types)
{
	std::vector<const rvsdg::type*> argument_types;
	for (const auto & arg : arguments)
		argument_types.push_back(&arg->type());

	test_op op(argument_types, result_types);
	return llvm::tac::create(op, arguments);
}

static inline std::vector<rvsdg::output*>
create_testop(
	rvsdg::region * region,
	const std::vector<rvsdg::output*> & operands,
	const std::vector<const rvsdg::type*> & result_types)
{
	std::vector<const rvsdg::type*> operand_types;
	for (const auto & operand : operands)
		operand_types.push_back(&operand->type());

	test_op op(operand_types, result_types);
	return rvsdg::simple_node::create_normalized(region, op, {operands});
}

}

#endif

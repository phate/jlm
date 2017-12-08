/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators/operators.hpp>
#include <jlm/ir/operators/sext.hpp>

namespace jlm {

/* setxt normal form */

sext_normal_form::~sext_normal_form()
{}

sext_normal_form::sext_normal_form(
	const std::type_info & opclass,
	jive::node_normal_form * parent,
	jive::graph * graph) noexcept
: unary_normal_form(opclass, parent, graph)
{}

/* sext operation */

static const jive_unop_reduction_path_t sext_reduction_bitunary = 128;
static const jive_unop_reduction_path_t sext_reduction_bitbinary = 129;

static bool
is_bitunary_reducible(const jive::output * operand)
{
	return jive::is_opnode<jive::bits::unary_op>(operand->node());
}

static bool
is_bitbinary_reducible(const jive::output * operand)
{
	return jive::is_opnode<jive::bits::binary_op>(operand->node());
}

static bool
is_inverse_reducible(const sext_op & op, const jive::output * operand)
{
	if (!operand->node())
		return false;

	auto top = dynamic_cast<const jlm::trunc_op*>(&operand->node()->operation());
	return top && top->nsrcbits() == op.ndstbits();
}

static jive::output *
perform_bitunary_reduction(const sext_op & op, jive::output * operand)
{
	JLM_DEBUG_ASSERT(is_bitunary_reducible(operand));
	auto unary = operand->node();
	auto uop = static_cast<const jive::bits::unary_op*>(&unary->operation());

	auto output = create_sext(op.ndstbits(), unary->input(0)->origin());
	return create_normalized(operand->region(), *uop->create(op.ndstbits()), {output})[0];
}

static jive::output *
perform_bitbinary_reduction(const sext_op & op, jive::output * operand)
{
	JLM_DEBUG_ASSERT(is_bitbinary_reducible(operand));
	auto binary = operand->node();
	auto bop = static_cast<const jive::bits::binary_op*>(&binary->operation());

	JLM_DEBUG_ASSERT(binary->ninputs() == 2);
	auto op1 = create_sext(op.ndstbits(), binary->input(0)->origin());
	auto op2 = create_sext(op.ndstbits(), binary->input(1)->origin());

	return create_normalized(operand->region(), *bop->create(op.ndstbits()), {op1, op2})[0];
}

static jive::output *
perform_inverse_reduction(const sext_op & op, jive::output * operand)
{
	JLM_DEBUG_ASSERT(is_inverse_reducible(op, operand));
	return operand->node()->input(0)->origin();
}

sext_op::~sext_op()
{}

bool
sext_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const sext_op*>(&other);
	return op && op->oport_ == oport_ && op->rport_ == rport_;
}

size_t
sext_op::narguments() const noexcept
{
	return 1;
}

const jive::port &
sext_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return oport_;
}

size_t
sext_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
sext_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return rport_;
}

std::string
sext_op::debug_string() const
{
	return strfmt("SEXT[", nsrcbits(), " -> ", ndstbits(), "]");
}

std::unique_ptr<jive::operation>
sext_op::copy() const
{
	return std::unique_ptr<jive::operation>(new sext_op(*this));
}

jive_unop_reduction_path_t
sext_op::can_reduce_operand(const jive::output * operand) const noexcept
{
	if (jive::is_bitconstant_node(producer(operand)))
		return jive_unop_reduction_constant;

	if (is_bitunary_reducible(operand))
		return sext_reduction_bitunary;

	if (is_bitbinary_reducible(operand))
		return sext_reduction_bitbinary;

	if (is_inverse_reducible(*this, operand))
		return jive_unop_reduction_inverse;

	return jive_unop_reduction_none;
}

jive::output *
sext_op::reduce_operand(
	jive_unop_reduction_path_t path,
	jive::output * operand) const
{
	if (path == jive_unop_reduction_constant) {
		auto c = static_cast<const jive::bits::constant_op*>(&producer(operand)->operation());
		return create_bitconstant(operand->region(), c->value().sext(ndstbits()-nsrcbits()));
	}

	if (path == sext_reduction_bitunary)
		return perform_bitunary_reduction(*this, operand);

	if (path == sext_reduction_bitbinary)
		return perform_bitbinary_reduction(*this, operand);

	if (path == jive_unop_reduction_inverse)
		return perform_inverse_reduction(*this, operand);

	return nullptr;
}

}

namespace {

static jive::node_normal_form *
create_sext_normal_form(
	const std::type_info & opclass,
	jive::node_normal_form * parent,
	jive::graph * graph)
{
	return new jlm::sext_normal_form(opclass, parent, graph);
}

static void __attribute__((constructor))
register_normal_form()
{
	jive::node_normal_form::register_factory(typeid(jlm::sext_op), create_sext_normal_form);
}

}

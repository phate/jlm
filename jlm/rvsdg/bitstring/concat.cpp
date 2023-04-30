/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/concat.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/bitstring/slice.hpp>
#include <jlm/rvsdg/reduction-helpers.hpp>

#include <string.h>

jive::output *
jive_bitconcat(const std::vector<jive::output*> & operands)
{
	std::vector<jive::bittype> types;
	for (const auto operand : operands)
		types.push_back(dynamic_cast<const jive::bittype&>(operand->type()));

	auto region = operands[0]->region();
	jive::bitconcat_op op(std::move(types));
	return jive::simple_node::create_normalized(region, op, {operands.begin(), operands.end()})[0];
}

namespace jive {

namespace {

jive::output *
concat_reduce_arg_pair(jive::output * arg1, jive::output * arg2)
{
	auto node1 = node_output::node(arg1);
	auto node2 = node_output::node(arg2);
	if (!node1 || !node2)
		return nullptr;

	auto arg1_constant = dynamic_cast<const bitconstant_op*>(&node1->operation());
	auto arg2_constant = dynamic_cast<const bitconstant_op*>(&node2->operation());
	if (arg1_constant && arg2_constant) {
		size_t nbits = arg1_constant->value().nbits() + arg2_constant->value().nbits();
		std::vector<char> bits(nbits);
		memcpy(&bits[0], &arg1_constant->value()[0], arg1_constant->value().nbits());
		memcpy(
			&bits[0] + arg1_constant->value().nbits(),
			&arg2_constant->value()[0],
			arg2_constant->value().nbits());

		std::string s(&bits[0], nbits);
		return create_bitconstant(node1->region(), s.c_str());
	}

	auto arg1_slice = dynamic_cast<const bitslice_op*>(&node1->operation());
	auto arg2_slice = dynamic_cast<const bitslice_op*>(&node2->operation());
	if (arg1_slice && arg2_slice && arg1_slice->high() == arg2_slice->low() &&
		node1->input(0)->origin() == node2->input(0)->origin()) {
		/* FIXME: support sign bit */
		return jive_bitslice(node1->input(0)->origin(), arg1_slice->low(), arg2_slice->high());
	}

	return nullptr;
}

std::vector<bittype>
types_from_arguments(const std::vector<jive::output*> & args)
{
	std::vector<bittype> types;
	for (const auto arg : args) {
		types.push_back(static_cast<const bittype&>(arg->type()));
	}
	return types;
}

}

class concat_normal_form final : public simple_normal_form {
public:
	virtual
	~concat_normal_form() noexcept;

	concat_normal_form(
		jive::node_normal_form * parent,
		jive::graph * graph)
		: simple_normal_form(typeid(bitconcat_op), parent, graph)
		, enable_reducible_(true)
		, enable_flatten_(true)
	{
	}

	virtual bool
	normalize_node(jive::node * node) const override
	{
		if (!get_mutable()) {
			return true;
		}

		auto args = operands(node);
		std::vector<jive::output*> new_args;

		/* possibly expand associative */
		if (get_flatten()) {
			new_args = base::detail::associative_flatten(
				args,
				[](jive::output * arg) {
					// FIXME: switch to comparing operator, not just typeid, after
					// converting "concat" to not be a binary operator anymore
					return is<bitconcat_op>(node_output::node(arg));
				});
		} else {
			new_args = args;
		}

		if (get_reducible()) {
			new_args = base::detail::pairwise_reduce(
				std::move(new_args),
				concat_reduce_arg_pair);

			if (new_args.size() == 1) {
				divert_users(node, new_args);
				remove(node);
				return false;
			}
		}
		
		if (args != new_args) {
			bitconcat_op op(types_from_arguments(new_args));
			divert_users(node, simple_node::create_normalized(node->region(), op, new_args));
			remove(node);
			return false;
		}

		return simple_normal_form::normalize_node(node);
	}

	virtual std::vector<jive::output*>
	normalized_create(
		jive::region * region,
		const jive::simple_op & op,
		const std::vector<jive::output*> & arguments) const override
	{
		std::vector<jive::output*> new_args;

		/* possibly expand associative */
		if (get_mutable() && get_flatten()) {
			new_args = base::detail::associative_flatten(
				arguments,
				[](jive::output * arg) {
					// FIXME: switch to comparing operator, not just typeid, after
					// converting "concat" to not be a binary operator anymore
					return is<bitconcat_op>(node_output::node(arg));
				});
		} else {
			new_args = arguments;
		}

		if (get_mutable() && get_reducible()) {
			new_args = base::detail::pairwise_reduce(
				std::move(new_args),
				concat_reduce_arg_pair);
			if (new_args.size() == 1)
				return new_args;
		}

		bitconcat_op new_op(types_from_arguments(new_args));
		return simple_normal_form::normalized_create(region, new_op, new_args);
	}

	virtual void
	set_reducible(bool enable)
	{
		if (get_reducible() == enable) {
			return;
		}

		children_set<concat_normal_form, &concat_normal_form::set_reducible>(enable);

		enable_reducible_ = enable;
		if (get_mutable() && enable)
			graph()->mark_denormalized();
	}
	inline bool
	get_reducible() const noexcept { return enable_reducible_; }

	virtual void
	set_flatten(bool enable)
	{
		if (get_flatten() == enable) {
			return;
		}

		children_set<concat_normal_form, &concat_normal_form::set_flatten>(enable);

		enable_flatten_ = enable;
		if (get_mutable() && enable)
			graph()->mark_denormalized();
	}
	inline bool
	get_flatten() const noexcept { return enable_flatten_; }

private:
	bool enable_reducible_;
	bool enable_flatten_;
};

concat_normal_form::~concat_normal_form() noexcept
{
}

static node_normal_form *
get_default_normal_form(
	const std::type_info & operator_class,
	jive::node_normal_form * parent,
	jive::graph * graph)
{
	return new concat_normal_form(parent, graph);
}

static void  __attribute__((constructor))
register_node_normal_form(void)
{
	jive::node_normal_form::register_factory(typeid(jive::bitconcat_op), get_default_normal_form);
}

bittype
bitconcat_op::aggregate_arguments(const std::vector<bittype> & types) noexcept
{
	size_t total = 0;
	for (const auto & t : types) {
		total += t.nbits();
	}
	return bittype(total);
}

std::vector<jive::port>
bitconcat_op::to_ports(const std::vector<bittype> & types)
{
	std::vector<jive::port> ports;
	for (const auto & type : types)
		ports.push_back({type});

	return ports;
}

bitconcat_op::~bitconcat_op() noexcept
{}

bool
bitconcat_op::operator==(const jive::operation & other) const noexcept
{
	auto op = dynamic_cast<const jive::bitconcat_op*>(&other);
	if (!op || op->narguments() != narguments())
		return false;

	for (size_t n = 0; n < narguments(); n++) {
		if (op->argument(n) != argument(n))
			return false;
	}

	return true;
}

jive_binop_reduction_path_t
bitconcat_op::can_reduce_operand_pair(
	const jive::output * arg1,
	const jive::output * arg2) const noexcept
{
	auto node1 = node_output::node(arg1);
	auto node2 = node_output::node(arg2);

	if (!node1 || !node2)
		return jive_binop_reduction_none;

	auto arg1_constant = is<bitconstant_op>(node1);
	auto arg2_constant = is<bitconstant_op>(node2);

	if (arg1_constant && arg2_constant) {
		return jive_binop_reduction_constants;
	}

	auto arg1_slice = dynamic_cast<const bitslice_op*>(&node1->operation());
	auto arg2_slice = dynamic_cast<const bitslice_op*>(&node2->operation());

	if (arg1_slice && arg2_slice){
		auto origin1 = node1->input(0)->origin();
		auto origin2 = node2->input(0)->origin();

		if (origin1 == origin2 && arg1_slice->high() == arg2_slice->low()) {
			return jive_binop_reduction_merge;
		}

		/* FIXME: support sign bit */
	}

	return jive_binop_reduction_none;
}

jive::output *
bitconcat_op::reduce_operand_pair(
	jive_binop_reduction_path_t path,
	jive::output * arg1,
	jive::output * arg2) const
{
	auto node1 = static_cast<node_output*>(arg1)->node();
	auto node2 = static_cast<node_output*>(arg2)->node();

	if (path == jive_binop_reduction_constants) {
		auto & arg1_constant = static_cast<const bitconstant_op&>(node1->operation());
		auto & arg2_constant = static_cast<const bitconstant_op&>(node2->operation());

		size_t nbits = arg1_constant.value().nbits() + arg2_constant.value().nbits();
		std::vector<char> bits(nbits);
		memcpy(&bits[0], &arg1_constant.value()[0], arg1_constant.value().nbits());
		memcpy(
			&bits[0] + arg1_constant.value().nbits(),
			&arg2_constant.value()[0],
			arg2_constant.value().nbits());

		return create_bitconstant(arg1->region(), &bits[0]);
	}

	if (path == jive_binop_reduction_merge) {
		auto arg1_slice = static_cast<const bitslice_op*>(&node1->operation());
		auto arg2_slice = static_cast<const bitslice_op*>(&node2->operation());
		return jive_bitslice(node1->input(0)->origin(), arg1_slice->low(), arg2_slice->high());

		/* FIXME: support sign bit */
	}

	return NULL;
}

enum jive::binary_op::flags
bitconcat_op::flags() const noexcept
{
	return binary_op::flags::associative;
}

std::string
bitconcat_op::debug_string() const
{
	return "BITCONCAT";
}

std::unique_ptr<jive::operation>
bitconcat_op::copy() const
{
	return std::unique_ptr<jive::operation>(new bitconcat_op(*this));
}

}

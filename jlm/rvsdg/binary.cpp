/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/binary.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/reduction-helpers.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include <deque>

namespace jive {

/* binary normal form */

namespace {

std::vector<jive::output*>
reduce_operands(const jive::binary_op & op, std::vector<jive::output*> args)
{
	/* pair-wise reduce */
	if (op.is_commutative()) {
		return base::detail::commutative_pairwise_reduce(
			std::move(args),
			[&op](jive::output * arg1, jive::output * arg2)
			{
				jive_binop_reduction_path_t reduction = op.can_reduce_operand_pair(arg1, arg2);
				return reduction != jive_binop_reduction_none
					? op.reduce_operand_pair(reduction, arg1, arg2)
					: nullptr;
			});
	} else {
		return base::detail::pairwise_reduce(
			std::move(args),
			[&op](jive::output * arg1, jive::output * arg2)
			{
				jive_binop_reduction_path_t reduction = op.can_reduce_operand_pair(arg1, arg2);
				return reduction != jive_binop_reduction_none
					? op.reduce_operand_pair(reduction, arg1, arg2)
					: nullptr;
			});
	}
}

}

binary_normal_form::~binary_normal_form() noexcept
{
}

binary_normal_form::binary_normal_form(
	const std::type_info & operator_class,
	jive::node_normal_form * parent,
	jive::graph * graph)
	: simple_normal_form(operator_class, parent, graph)
	, enable_reducible_(true)
	, enable_reorder_(true)
	, enable_flatten_(true)
	, enable_distribute_(true)
	, enable_factorize_(true)
{
	if (auto p = dynamic_cast<binary_normal_form *>(parent)) {
		enable_reducible_ = p->enable_reducible_;
		enable_reorder_ = p->enable_reorder_;
		enable_flatten_ = p->enable_flatten_;
		enable_distribute_ = p->enable_distribute_;
		enable_factorize_ = p->enable_factorize_;
	}
}

bool
binary_normal_form::normalize_node(jive::node * node) const
{
	const jive::operation & base_op = node->operation();
	const auto & op = *static_cast<const jive::binary_op*>(&base_op);

	return normalize_node(node, op);
}

bool
binary_normal_form::normalize_node(jive::node * node, const binary_op & op) const
{
	if (!get_mutable()) {
		return true;
	}

	auto args = operands(node);
	std::vector<jive::output*> new_args;

	/* possibly expand associative */
	if (get_flatten() && op.is_associative()) {
		new_args = base::detail::associative_flatten(
			args,
			[&op](jive::output * arg) {
				if (!is<node_output>(arg))
					return false;

				auto node = static_cast<node_output*>(arg)->node();
				auto fb_op = dynamic_cast<const flattened_binary_op*>(&node->operation());
				return node->operation() == op || (fb_op && fb_op->bin_operation() == op);
			});
	} else {
		new_args = args;
	}

	if (get_reducible()) {
		auto tmp = reduce_operands(op, std::move(new_args));
		new_args = {tmp.begin(), tmp.end()};

		if (new_args.size() == 1) {
			node->output(0)->divert_users(new_args[0]);
			node->region()->remove_node(node);
			return false;
		}
	}

	/* FIXME: reorder for commutative operation */

	/* FIXME: attempt distributive transform */

	bool changes = (args != new_args);

	if (changes) {
		std::unique_ptr<simple_op> tmp_op;
		if (new_args.size() > 2)
			tmp_op.reset(new flattened_binary_op(op, new_args.size()));

		JIVE_DEBUG_ASSERT(new_args.size() >= 2);
		const auto & new_op = tmp_op ? *tmp_op : static_cast<const simple_op&>(op);
		divert_users(node, simple_node::create_normalized(node->region(), new_op, new_args));
		remove(node);
		return false;
	}

	return simple_normal_form::normalize_node(node);
}

std::vector<jive::output*>
binary_normal_form::normalized_create(
	jive::region * region,
	const jive::simple_op & base_op,
	const std::vector<jive::output*> & args) const
{
	const auto & op = *static_cast<const jive::binary_op*>(&base_op);

	std::vector<jive::output*> new_args(args.begin(), args.end());

	/* possibly expand associative */
	if (get_mutable() && get_flatten() && op.is_associative()) {
		new_args = base::detail::associative_flatten(
			args,
			[&op](jive::output* arg) {
				if (!is<node_output>(arg))
					return false;

				auto node = static_cast<node_output*>(arg)->node();
				auto fb_op = dynamic_cast<const flattened_binary_op*>(&node->operation());
				return node->operation() == op || (fb_op && fb_op->bin_operation() == op);
			});
	}

	if (get_mutable() && get_reducible()) {
		new_args = reduce_operands(op, std::move(new_args));
		if (new_args.size() == 1)
			return new_args;
	}

	/* FIXME: reorder for commutative operation */

	/* FIXME: attempt distributive transform */
	std::unique_ptr<simple_op> tmp_op;
	if (new_args.size() > 2) {
		tmp_op.reset(new flattened_binary_op(op, new_args.size()));
	}

	region = new_args[0]->region();
	const auto & new_op = tmp_op ? *tmp_op : static_cast<const simple_op&>(op);
	return simple_normal_form::normalized_create(region, new_op, new_args);
}

void
binary_normal_form::set_reducible(bool enable)
{
	if (get_reducible() == enable) {
		return;
	}

	children_set<binary_normal_form, &binary_normal_form::set_reducible>(enable);

	enable_reducible_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

void
binary_normal_form::set_flatten(bool enable)
{
	if (get_flatten() == enable) {
		return;
	}

	children_set<binary_normal_form, &binary_normal_form::set_flatten>(enable);

	enable_flatten_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

void
binary_normal_form::set_reorder(bool enable)
{
	if (get_reorder() == enable) {
		return;
	}

	children_set<binary_normal_form, &binary_normal_form::set_reorder>(enable);

	enable_reorder_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

void
binary_normal_form::set_distribute(bool enable)
{
	if (get_distribute() == enable) {
		return;
	}

	children_set<binary_normal_form, &binary_normal_form::set_distribute>(enable);

	enable_distribute_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

void
binary_normal_form::set_factorize(bool enable)
{
	if (get_factorize() == enable) {
		return;
	}

	children_set<binary_normal_form, &binary_normal_form::set_factorize>(enable);

	enable_factorize_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

/* flattened binary normal form */

flattened_binary_normal_form::~flattened_binary_normal_form() noexcept
{
}

flattened_binary_normal_form::flattened_binary_normal_form(
	const std::type_info & operator_class,
	jive::node_normal_form * parent,
	jive::graph * graph)
	: simple_normal_form(operator_class, parent, graph)
{
}

bool
flattened_binary_normal_form::normalize_node(jive::node * node) const
{
	const auto & op = static_cast<const flattened_binary_op&>(node->operation());
	const auto & bin_op = op.bin_operation();
	auto nf = graph()->node_normal_form(typeid(bin_op));

	return static_cast<const binary_normal_form *>(nf)->normalize_node(node, bin_op);
}

std::vector<jive::output*>
flattened_binary_normal_form::normalized_create(
	jive::region * region,
	const jive::simple_op & base_op,
	const std::vector<jive::output*> & arguments) const
{
	const auto & op = static_cast<const flattened_binary_op&>(base_op);
	const auto & bin_op = op.bin_operation();

	auto nf = static_cast<const binary_normal_form*>(
		graph()->node_normal_form(typeid(bin_op)));
	return nf->normalized_create(region, bin_op, arguments);
}

/* binary operator */

binary_op::~binary_op() noexcept
{}

enum jive::binary_op::flags
binary_op::flags() const noexcept
{
	return jive::binary_op::flags::none;
}

/* flattened binary operator */

flattened_binary_op::~flattened_binary_op() noexcept
{
}

bool
flattened_binary_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const flattened_binary_op*>(&other);
	return op
	    && op->bin_operation() == bin_operation()
	    && op->narguments() == narguments();
}

std::string
flattened_binary_op::debug_string() const
{
	return detail::strfmt("FLATTENED[", op_->debug_string(),"]");
}

std::unique_ptr<jive::operation>
flattened_binary_op::copy() const
{
	std::unique_ptr<binary_op> copied_op(static_cast<binary_op *>(op_->copy().release()));
	return std::unique_ptr<jive::operation>(
		new flattened_binary_op(std::move(copied_op), narguments()));
}

/*
	FIXME: The reduce_parallel and reduce_linear functions only differ in where they add
	the new output to the working list. Unify both functions.
*/

static jive::output *
reduce_parallel(
	const binary_op & op,
	const std::vector<jive::output*> & operands)
{
	JIVE_DEBUG_ASSERT(operands.size() > 1);
	auto region = operands.front()->region();
	JIVE_DEBUG_ASSERT(binary_op::normal_form(region->graph())->get_flatten() == false);

	std::deque<jive::output*> worklist(operands.begin(), operands.end());
	while (worklist.size() > 1) {
		auto op1 = worklist.front();
		worklist.pop_front();
		auto op2 = worklist.front();
		worklist.pop_front();

		auto output = simple_node::create_normalized(region, op, {op1, op2})[0];
		worklist.push_back(output);
	}

	JIVE_DEBUG_ASSERT(worklist.size() == 1);
	return worklist.front();
}

static jive::output *
reduce_linear(
	const binary_op & op,
	const std::vector<jive::output*> & operands)
{
	JIVE_DEBUG_ASSERT(operands.size() > 1);
	auto region = operands.front()->region();
	JIVE_DEBUG_ASSERT(binary_op::normal_form(region->graph())->get_flatten() == false);

	std::deque<jive::output*> worklist(operands.begin(), operands.end());
	while (worklist.size() > 1) {
		auto op1 = worklist.front();
		worklist.pop_front();
		auto op2 = worklist.front();
		worklist.pop_front();

		auto output = simple_node::create_normalized(region, op, {op1, op2})[0];
		worklist.push_front(output);
	}

	JIVE_DEBUG_ASSERT(worklist.size() == 1);
	return worklist.front();
}

jive::output *
flattened_binary_op::reduce(
	const flattened_binary_op::reduction & reduction,
	const std::vector<jive::output*> & operands) const
{
	JIVE_DEBUG_ASSERT(operands.size() > 1);
	auto graph = operands[0]->region()->graph();

	static std::unordered_map<
		flattened_binary_op::reduction,
		std::function<jive::output*(const binary_op&, const std::vector<jive::output*>&)>
	> map({
	  {reduction::linear, reduce_linear}
	, {reduction::parallel, reduce_parallel}
	});

	binary_op::normal_form(graph)->set_flatten(false);
	JIVE_DEBUG_ASSERT(map.find(reduction) != map.end());
	return map[reduction](bin_operation(), operands);
}

void
flattened_binary_op::reduce(
	jive::region * region,
	const flattened_binary_op::reduction & reduction)
{
	for (auto & node : topdown_traverser(region)) {
		if (is<flattened_binary_op>(node)) {
			auto op = static_cast<const flattened_binary_op*>(&node->operation());
			auto output = op->reduce(reduction, operands(node));
			node->output(0)->divert_users(output);
			remove(node);
		} else if (auto structnode = dynamic_cast<const structural_node*>(node)) {
			for (size_t n = 0; n < structnode->nsubregions(); n++)
				reduce(structnode->subregion(n), reduction);
		}
	}

	JIVE_DEBUG_ASSERT(!region::Contains<flattened_binary_op>(*region, true));
}

}

/* node class */

/* node class inheritable methods */

jive::node_normal_form *
jive_binary_operation_get_default_normal_form_(
	const std::type_info & operator_class,
	jive::node_normal_form * parent,
	jive::graph * graph)
{
	jive::binary_normal_form * nf = new jive::binary_normal_form(operator_class,  parent, graph);

	return nf;
}

jive::node_normal_form *
jive_flattened_binary_operation_get_default_normal_form_(
	const std::type_info & operator_class,
	jive::node_normal_form * parent,
	jive::graph * graph)
{
	return new jive::flattened_binary_normal_form(operator_class,  parent, graph);
}

static void  __attribute__((constructor))
register_node_normal_form(void)
{
	jive::node_normal_form::register_factory(
		typeid(jive::binary_op), jive_binary_operation_get_default_normal_form_);
	jive::node_normal_form::register_factory(
		typeid(jive::flattened_binary_op), jive_flattened_binary_operation_get_default_normal_form_);
}

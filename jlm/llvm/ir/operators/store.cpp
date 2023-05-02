/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/store.hpp>

namespace jlm {

StoreOperation::~StoreOperation() noexcept
= default;

bool
StoreOperation::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const StoreOperation*>(&other);
  return op
      && op->NumStates() == NumStates()
      && op->GetPointerType() == GetPointerType()
      && op->GetAlignment() == GetAlignment();
}

std::string
StoreOperation::debug_string() const
{
	return "STORE";
}

std::unique_ptr<jive::operation>
StoreOperation::copy() const
{
	return std::unique_ptr<jive::operation>(new StoreOperation(*this));
}

/* store normal form */

static bool
is_store_mux_reducible(const std::vector<jive::output*> & operands)
{
	JLM_ASSERT(operands.size() > 2);

	auto memStateMergeNode = jive::node_output::node(operands[2]);
	if (!is<MemStateMergeOperator>(memStateMergeNode))
		return false;

	for (size_t n = 2; n < operands.size(); n++) {
		auto node = jive::node_output::node(operands[n]);
		if (node != memStateMergeNode)
			return false;
	}

	return true;
}

static bool
is_store_store_reducible(
	const StoreOperation & op,
	const std::vector<jive::output*> & operands)
{
	JLM_ASSERT(operands.size() > 2);

	auto storenode = jive::node_output::node(operands[2]);
	if (!is<StoreOperation>(storenode))
		return false;

	if (op.NumStates() != storenode->noutputs())
		return false;

	/* check for same address */
	if (operands[0] != storenode->input(0)->origin())
		return false;

	for (size_t n = 2; n < operands.size(); n++) {
		if (jive::node_output::node(operands[n]) != storenode || operands[n]->nusers() != 1)
			return false;
	}

	auto other = static_cast<const StoreOperation*>(&storenode->operation());
	JLM_ASSERT(op.GetAlignment() == other->GetAlignment());
	return true;
}

static bool
is_store_alloca_reducible(const std::vector<jive::output*> & operands)
{
	if (operands.size() == 3)
		return false;

	auto alloca = jive::node_output::node(operands[0]);
	if (!alloca || !is<alloca_op>(alloca->operation()))
		return false;

	std::unordered_set<jive::output*> states(std::next(std::next(operands.begin())), operands.end());
	if (states.find(alloca->output(1)) == states.end())
		return false;

	if (alloca->output(1)->nusers() != 1)
		return false;

	return true;
}

static bool
is_multiple_origin_reducible(const std::vector<jive::output*> & operands)
{
	std::unordered_set<jive::output*> states(std::next(std::next(operands.begin())), operands.end());
	return states.size() != operands.size()-2;
}

static std::vector<jive::output*>
perform_store_mux_reduction(
	const StoreOperation & op,
	const std::vector<jive::output*> & operands)
{
	auto memStateMergeNode = jive::node_output::node(operands[2]);
	auto memStateMergeOperands = jive::operands(memStateMergeNode);

	auto states = StoreNode::Create(operands[0], operands[1], memStateMergeOperands, op.GetAlignment());
	return {MemStateMergeOperator::Create(states)};
}

static std::vector<jive::output*>
perform_store_store_reduction(
	const StoreOperation & op,
	const std::vector<jive::output*> & operands)
{
	JLM_ASSERT(is_store_store_reducible(op, operands));
	auto storenode = jive::node_output::node(operands[2]);

	auto storeops = jive::operands(storenode);
	std::vector<jive::output*> states(std::next(std::next(storeops.begin())), storeops.end());
	return StoreNode::Create(operands[0], operands[1], states, op.GetAlignment());
}

static std::vector<jive::output*>
perform_store_alloca_reduction(
	const StoreOperation & op,
	const std::vector<jive::output*> & operands)
{
	auto value = operands[1];
	auto address = operands[0];
	auto alloca_state = jive::node_output::node(address)->output(1);
	std::unordered_set<jive::output*> states(std::next(std::next(operands.begin())), operands.end());

	auto outputs = StoreNode::Create(address, value, {alloca_state}, op.GetAlignment());
	states.erase(alloca_state);
	states.insert(outputs[0]);
	return {states.begin(), states.end()};
}

static std::vector<jive::output*>
perform_multiple_origin_reduction(
	const StoreOperation & op,
	const std::vector<jive::output*> & operands)
{
	std::unordered_set<jive::output*> states(std::next(std::next(operands.begin())), operands.end());
	return StoreNode::Create(operands[0], operands[1], {states.begin(), states.end()},
                                op.GetAlignment());
}

store_normal_form::~store_normal_form()
{}

store_normal_form::store_normal_form(
	const std::type_info & opclass,
	jive::node_normal_form * parent,
	jive::graph * graph) noexcept
: simple_normal_form(opclass, parent, graph)
, enable_store_mux_(false)
, enable_store_store_(false)
, enable_store_alloca_(false)
, enable_multiple_origin_(false)
{
	if (auto p = dynamic_cast<const store_normal_form*>(parent)) {
		enable_multiple_origin_ = p->enable_multiple_origin_;
		enable_store_store_ = p->enable_store_store_;
		enable_store_mux_ = p->enable_store_mux_;
	}
}

bool
store_normal_form::normalize_node(jive::node * node) const
{
	JLM_ASSERT(is<StoreOperation>(node->operation()));
	auto op = static_cast<const jlm::StoreOperation*>(&node->operation());
	auto operands = jive::operands(node);

	if (!get_mutable())
		return true;

	if (get_store_mux_reducible() && is_store_mux_reducible(operands)) {
		divert_users(node, perform_store_mux_reduction(*op, operands));
		node->region()->remove_node(node);
		return false;
	}

	if (get_store_store_reducible() && is_store_store_reducible(*op, operands)) {
		divert_users(node, perform_store_store_reduction(*op, operands));
		remove(node);
		return false;
	}

	if (get_store_alloca_reducible() && is_store_alloca_reducible(operands)) {
		divert_users(node, perform_store_alloca_reduction(*op, operands));
		node->region()->remove_node(node);
		return false;
	}

	if (get_multiple_origin_reducible() && is_multiple_origin_reducible(operands)) {
		auto outputs = perform_multiple_origin_reduction(*op, operands);
		auto new_node = jive::node_output::node(outputs[0]);

		std::unordered_map<jive::output*, jive::output*> origin2output;
		for (size_t n = 0; n < outputs.size(); n++) {
			auto origin = new_node->input(n+2)->origin();
			JLM_ASSERT(origin2output.find(origin) == origin2output.end());
			origin2output[origin] = outputs[n];
		}

		for (size_t n = 2; n < node->ninputs(); n++) {
			auto origin = node->input(n)->origin();
			JLM_ASSERT(origin2output.find(origin) != origin2output.end());
			node->output(n-2)->divert_users(origin2output[origin]);
		}
		remove(node);
		return false;
	}

	return simple_normal_form::normalize_node(node);
}

std::vector<jive::output*>
store_normal_form::normalized_create(
	jive::region * region,
	const jive::simple_op & op,
	const std::vector<jive::output*> & ops) const
{
	JLM_ASSERT(is<StoreOperation>(op));
	auto sop = static_cast<const jlm::StoreOperation*>(&op);

	if (!get_mutable())
		return simple_normal_form::normalized_create(region, op, ops);

	auto operands = ops;
	if (get_store_mux_reducible() && is_store_mux_reducible(operands))
		return perform_store_mux_reduction(*sop, operands);

	if (get_store_alloca_reducible() && is_store_alloca_reducible(operands))
		return perform_store_alloca_reduction(*sop, operands);

	if (get_multiple_origin_reducible() && is_multiple_origin_reducible(operands))
		return perform_multiple_origin_reduction(*sop, operands);

	return simple_normal_form::normalized_create(region, op, operands);
}

void
store_normal_form::set_store_mux_reducible(bool enable)
{
	if (get_store_mux_reducible() == enable)
		return;

	children_set<store_normal_form, &store_normal_form::set_store_mux_reducible>(enable);

	enable_store_mux_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

void
store_normal_form::set_store_store_reducible(bool enable)
{
	if (get_store_store_reducible() == enable)
		return;

	children_set<store_normal_form, &store_normal_form::set_store_store_reducible>(enable);

	enable_store_store_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

void
store_normal_form::set_store_alloca_reducible(bool enable)
{
	if (get_store_alloca_reducible() == enable)
		return;

	children_set<store_normal_form, &store_normal_form::set_store_alloca_reducible>(enable);

	enable_store_alloca_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

void
store_normal_form::set_multiple_origin_reducible(bool enable)
{
	if (get_multiple_origin_reducible() == enable)
		return;

	children_set<store_normal_form, &store_normal_form::set_multiple_origin_reducible>(enable);

	enable_multiple_origin_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

}

namespace {

static jive::node_normal_form *
create_store_normal_form(
	const std::type_info & opclass,
	jive::node_normal_form * parent,
	jive::graph * graph)
{
	return new jlm::store_normal_form(opclass, parent, graph);
}

static void __attribute__((constructor))
register_normal_form()
{
	jive::node_normal_form::register_factory(typeid(jlm::StoreOperation), create_store_normal_form);
}

}

/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/jlm/ir/operators/operators.hpp>
#include <jlm/jlm/ir/operators/store.hpp>
#include <jlm/jlm/opt/push.hpp>

#include <jive/rvsdg/gamma.h>
#include <jive/rvsdg/theta.h>
#include <jive/rvsdg/traverser.h>

#include <deque>

#ifdef PSHTIME
#include <chrono>
#include <iostream>
#endif

namespace jlm {

class worklist {
public:
	inline void
	push_back(jive::node * node) noexcept
	{
		if (set_.find(node) != set_.end())
			return;

		queue_.push_back(node);
		set_.insert(node);
	}

	inline jive::node *
	pop_front() noexcept
	{
		JLM_DEBUG_ASSERT(!empty());
		auto node = queue_.front();
		queue_.pop_front();
		set_.erase(node);
		return node;
	}

	inline bool
	empty() const noexcept
	{
		JLM_DEBUG_ASSERT(queue_.size() == set_.size());
		return queue_.empty();
	}

private:
	std::deque<jive::node*> queue_;
	std::unordered_set<jive::node*> set_;
};

static bool
has_side_effects(const jive::node * node)
{
	for (size_t n = 0; n < node->noutputs(); n++) {
		if (dynamic_cast<const jive::statetype*>(&node->output(n)->type()))
			return true;
	}

	return false;
}

static std::vector<jive::argument*>
copy_from_gamma(jive::node * node, size_t r)
{
	JLM_DEBUG_ASSERT(jive::is<jive::gamma_op>(node->region()->node()));
	JLM_DEBUG_ASSERT(node->depth() == 0);

	auto target = node->region()->node()->region();
	auto gamma = static_cast<jive::gamma_node*>(node->region()->node());

	std::vector<jive::output*> operands;
	for (size_t n = 0; n < node->ninputs(); n++) {
		JLM_DEBUG_ASSERT(dynamic_cast<const jive::argument*>(node->input(n)->origin()));
		auto argument = static_cast<const jive::argument*>(node->input(n)->origin());
		operands.push_back(argument->input()->origin());
	}

	std::vector<jive::argument*> arguments;
	auto copy = node->copy(target, operands);
	for (size_t n = 0; n < copy->noutputs(); n++) {
		auto ev = gamma->add_entryvar(copy->output(n));
		node->output(n)->divert_users(ev->argument(r));
		arguments.push_back(ev->argument(r));
	}

	return arguments;
}

static std::vector<jive::argument*>
copy_from_theta(jive::node * node)
{
	JLM_DEBUG_ASSERT(jive::is<jive::theta_op>(node->region()->node()));
	JLM_DEBUG_ASSERT(node->depth() == 0);

	auto target = node->region()->node()->region();
	auto theta = static_cast<jive::theta_node*>(node->region()->node());

	std::vector<jive::output*> operands;
	for (size_t n = 0; n < node->ninputs(); n++) {
		JLM_DEBUG_ASSERT(dynamic_cast<const jive::argument*>(node->input(n)->origin()));
		auto argument = static_cast<const jive::argument*>(node->input(n)->origin());
		operands.push_back(argument->input()->origin());
	}

	std::vector<jive::argument*> arguments;
	auto copy = node->copy(target, operands);
	for (size_t n = 0; n < copy->noutputs(); n++) {
		auto lv = theta->add_loopvar(copy->output(n));
		node->output(n)->divert_users(lv->argument());
		arguments.push_back(lv->argument());
	}

	return arguments;
}

void
push(jive::gamma_node * gamma)
{
	for (size_t r = 0; r < gamma->nsubregions(); r++) {
		auto region = gamma->subregion(r);

		/* push out all nullary nodes */
		for (auto & node : region->top_nodes) {
			if (!has_side_effects(&node))
				copy_from_gamma(&node, r);
		}

		/* initialize worklist */
		worklist wl;
		for (size_t n = 0; n < region->narguments(); n++) {
			auto argument = region->argument(n);
			for (const auto & user : *argument) {
				if (user->node() && user->node()->depth() == 0)
					wl.push_back(user->node());
			}
		}

		/* process worklist */
		while (!wl.empty()) {
			auto node = wl.pop_front();

			/* we cannot push out nodes with side-effects */
			if (has_side_effects(node))
				continue;

			auto arguments = copy_from_gamma(node, r);

			/* add consumers to worklist */
			for (const auto & argument : arguments) {
				for (const auto & user : *argument) {
					if (user->node() && user->node()->depth() == 0)
						wl.push_back(user->node());
				}
			}
		}
	}
}

static bool
is_theta_invariant(
	const jive::node * node,
	const std::unordered_set<jive::argument*> & invariants)
{
	JLM_DEBUG_ASSERT(jive::is<jive::theta_op>(node->region()->node()));
	JLM_DEBUG_ASSERT(node->depth() == 0);

	for (size_t n = 0; n < node->ninputs(); n++) {
		JLM_DEBUG_ASSERT(dynamic_cast<const jive::argument*>(node->input(n)->origin()));
		auto argument = static_cast<jive::argument*>(node->input(n)->origin());
		if (invariants.find(argument) == invariants.end())
			return false;
	}

	return true;
}

void
push_top(jive::theta_node * theta)
{
	auto subregion = theta->subregion();

	/* push out all nullary nodes */
	for (auto & node : subregion->top_nodes) {
		if (!has_side_effects(&node))
			copy_from_theta(&node);
	}

	/* collect loop invariant arguments */
	std::unordered_set<jive::argument*> invariants;
	for (const auto & lv : *theta) {
		if (lv->result()->origin() == lv->argument())
			invariants.insert(lv->argument());
	}

	/* initialize worklist */
	worklist wl;
	for (const auto & lv : *theta) {
		auto argument = lv->argument();
		for (const auto & user : *argument) {
			if (user->node() && user->node()->depth() == 0
			&& is_theta_invariant(user->node(), invariants))
				wl.push_back(user->node());
		}
	}

	/* process worklist */
	while (!wl.empty()) {
		auto node = wl.pop_front();

		/* we cannot push out nodes with side-effects */
		if (has_side_effects(node))
			continue;

		auto arguments = copy_from_theta(node);
		invariants.insert(arguments.begin(), arguments.end());

		/* add consumers to worklist */
		for (const auto & argument : arguments) {
			for (const auto  & user : *argument) {
				if (user->node() && user->node()->depth() == 0
				&& is_theta_invariant(user->node(), invariants))
					wl.push_back(user->node());
			}
		}
	}
}

static bool
is_invariant(const jive::argument * argument)
{
	JLM_DEBUG_ASSERT(jive::is<jive::theta_op>(argument->region()->node()));
	return argument->region()->result(argument->index()+1)->origin() == argument;
}

static bool
is_movable_store(jive::node * node)
{
	JLM_DEBUG_ASSERT(jive::is<jive::theta_op>(node->region()->node()));
	JLM_DEBUG_ASSERT(jive::is<store_op>(node));

	auto address = dynamic_cast<jive::argument*>(node->input(0)->origin());
	if (!address || !is_invariant(address) || address->nusers() != 2)
		return false;

	for (size_t n = 2; n < node->ninputs(); n++) {
		auto argument = dynamic_cast<jive::argument*>(node->input(n)->origin());
		if (!argument || argument->nusers() > 1)
			return false;
	}

	for (size_t n = 0; n < node->noutputs(); n++) {
		auto output = node->output(n);
		if (output->nusers() != 1)
			return false;

		if (!dynamic_cast<jive::result*>(*output->begin()))
			return false;
	}

	return true;
}

static void
pushout_store(jive::node * storenode)
{
	JLM_DEBUG_ASSERT(jive::is<jive::theta_op>(storenode->region()->node()));
	JLM_DEBUG_ASSERT(jive::is<store_op>(storenode) && is_movable_store(storenode));
	auto theta = static_cast<jive::theta_node*>(storenode->region()->node());
	auto storeop = static_cast<const jlm::store_op*>(&storenode->operation());
	auto oaddress = static_cast<jive::argument*>(storenode->input(0)->origin());
	auto ovalue = storenode->input(1)->origin();

	/* insert new value for store */
	auto nvalue = theta->add_loopvar(undef_constant_op::create(theta->region(), ovalue->type()));
	nvalue->result()->divert_to(ovalue);

	/* collect store operands */
	std::vector<jive::output*> states;
	auto address = oaddress->input()->origin();
	for (size_t n = 0; n < storenode->noutputs(); n++) {
		JLM_DEBUG_ASSERT(storenode->output(n)->nusers() == 1);
		auto result = static_cast<jive::result*>(*storenode->output(n)->begin());
		result->divert_to(storenode->input(n+2)->origin());
		states.push_back(result->output());
	}

	/* create new store and redirect theta output users */
	auto nstates = jlm::create_store(address, nvalue, states, storeop->alignment());
	for (size_t n = 0; n < states.size(); n++) {
		std::unordered_set<jive::input*> users;
		for (const auto & user : *states[n]) {
			if (user->node() != nstates[0]->node())
				users.insert(user);
		}

		for (const auto & user : users)
			user->divert_to(nstates[n]);
	}

	remove(storenode);
}

void
push_bottom(jive::theta_node * theta)
{
	for (const auto & lv : *theta) {
		auto storenode = lv->result()->origin()->node();
		if (jive::is<store_op>(storenode) && is_movable_store(storenode)) {
			pushout_store(storenode);
			break;
		}
	}
}

void
push(jive::theta_node * theta)
{
	bool done = false;
	while (!done) {
		auto nnodes = theta->subregion()->nnodes();
		push_top(theta);
		push_bottom(theta);
		if (nnodes == theta->subregion()->nnodes())
			done = true;
	}
}

static void
push(jive::region * region)
{
	for (auto node : jive::topdown_traverser(region)) {
		if (auto strnode = dynamic_cast<const jive::structural_node*>(node)) {
			for (size_t n = 0; n < strnode->nsubregions(); n++)
				push(strnode->subregion(n));
		}

		if (auto gamma = dynamic_cast<jive::gamma_node*>(node))
			push(gamma);

		if (auto theta = dynamic_cast<jive::theta_node*>(node))
			push(theta);
	}
}

void
push(jive::graph & graph)
{
	auto root = graph.root();

	#ifdef PSHTIME
		auto nnodes = jive::nnodes(root);
		auto start = std::chrono::high_resolution_clock::now();
	#endif

	push(root);

	#ifdef PSHTIME
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "PSHTIME: "
		          << nnodes
		          << " " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()
		          << "\n";
	#endif
}

}

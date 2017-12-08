/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/opt/push.hpp>

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
	JLM_DEBUG_ASSERT(is_gamma_node(node->region()->node()));
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
		node->output(n)->replace(ev->argument(r));
		arguments.push_back(ev->argument(r));
	}

	return arguments;
}

static std::vector<jive::argument*>
copy_from_theta(jive::node * node)
{
	JLM_DEBUG_ASSERT(is_theta_node(node->region()->node()));
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
		node->output(n)->replace(lv->argument());
		arguments.push_back(lv->argument());
	}

	return arguments;
}

static void
gamma_push(jive::structural_node * gamma)
{
	JLM_DEBUG_ASSERT(is_gamma_op(gamma->operation()));

	for (size_t r = 0; r < gamma->nsubregions(); r++) {
		auto region = gamma->subregion(r);

		/* push out all nullary nodes */
		jive::node * node;
		JIVE_LIST_ITERATE(region->top_nodes, node, region_top_node_list) {
			if (has_side_effects(node))
				continue;

			copy_from_gamma(node, r);
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
	JLM_DEBUG_ASSERT(is_theta_node(node->region()->node()));
	JLM_DEBUG_ASSERT(node->depth() == 0);

	for (size_t n = 0; n < node->ninputs(); n++) {
		JLM_DEBUG_ASSERT(dynamic_cast<const jive::argument*>(node->input(n)->origin()));
		auto argument = static_cast<jive::argument*>(node->input(n)->origin());
		if (invariants.find(argument) == invariants.end())
			return false;
	}

	return true;
}

static void
theta_push(jive::theta_node * theta)
{
	auto subregion = theta->subregion();

	/* push out all nullary nodes */
	jive::node * node;
	JIVE_LIST_ITERATE(subregion->top_nodes, node, region_top_node_list) {
		if (has_side_effects(node))
			continue;

		copy_from_theta(node);
	}

	/* collect loop invariant arguments */
	std::unordered_set<jive::argument*> invariants;
	for (const auto & lv : *theta) {
		if (lv.result()->origin() == lv.argument())
			invariants.insert(lv.argument());
	}

	/* initialize worklist */
	worklist wl;
	for (const auto & lv : *theta) {
		auto argument = lv.argument();
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

static void
push(jive::region * region)
{
	for (auto node : jive::topdown_traverser(region)) {
		if (auto strnode = dynamic_cast<const jive::structural_node*>(node)) {
			for (size_t n = 0; n < strnode->nsubregions(); n++)
				push(strnode->subregion(n));
		}

		if (is_gamma_op(node->operation()))
			gamma_push(static_cast<jive::structural_node*>(node));

		if (auto theta = dynamic_cast<jive::theta_node*>(node))
			theta_push(theta);
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

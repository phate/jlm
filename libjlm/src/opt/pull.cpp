/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/rvsdg/gamma.h>
#include <jive/rvsdg/traverser.h>

#include <jlm/common.hpp>
#include <jlm/ir/rvsdg.hpp>
#include <jlm/opt/pull.hpp>

namespace jlm {

static bool
empty(const jive::gamma_node * gamma)
{
	for (size_t n = 0; n < gamma->nsubregions(); n++) {
		if (gamma->subregion(n)->nnodes() != 0)
			return false;
	}

	return true;
}

static bool
single_successor(const jive::node * node)
{
	std::unordered_set<jive::node*> successors;
	for (size_t n = 0; n < node->noutputs(); n++) {
		for (const auto & user : *node->output(n))
			successors.insert(user->node());
	}

	return successors.size() == 1;
}

static void
remove(jive::gamma_input * input)
{
	auto gamma = input->node();

	for (size_t n = 0; n < gamma->nsubregions(); n++)
		gamma->subregion(n)->remove_argument(input->index()-1);
	gamma->remove_input(input->index());
}

static void
pullin_node(jive::gamma_node * gamma, jive::node * node)
{
	/* collect operands */
	std::vector<std::vector<jive::output*>> operands(gamma->nsubregions());
	for (size_t i = 0; i < node->ninputs(); i++) {
		auto ev = gamma->add_entryvar(node->input(i)->origin());
		for (size_t a = 0; a < ev->narguments(); a++)
			operands[a].push_back(ev->argument(a));
	}

	/* copy node into subregions */
	for (size_t r = 0; r < gamma->nsubregions(); r++) {
		auto copy = node->copy(gamma->subregion(r), operands[r]);

		/* redirect outputs */
		for (size_t o = 0; o < node->noutputs(); o++) {
			for (const auto & user : *node->output(o)) {
				JLM_DEBUG_ASSERT(dynamic_cast<jive::structural_input*>(user));
				auto sinput = static_cast<jive::structural_input*>(user);
				auto argument = gamma->subregion(r)->argument(sinput->index()-1);
				argument->divert_users(copy->output(o));
			}
		}
	}
}

static void
cleanup(jive::gamma_node * gamma, jive::node * node)
{
	JLM_DEBUG_ASSERT(single_successor(node));

	/* remove entry variables and node */
	for (size_t n = 0; n < node->noutputs(); n++) {
		while (node->output(n)->nusers() != 0)
			remove(static_cast<jive::gamma_input*>(*node->output(n)->begin()));
	}
	remove(node);
}

void
pullin_top(jive::gamma_node * gamma)
{
	/* FIXME: This is inefficient. We can do better. */
	auto ev = gamma->begin_entryvar();
	while (ev != gamma->end_entryvar()) {
		auto node = ev->origin()->node();
		if (node && gamma->predicate()->origin()->node() != node && single_successor(node)) {
			pullin_node(gamma, node);

			cleanup(gamma, node);

			ev = gamma->begin_entryvar();
		} else {
			ev++;
		}
	}
}

void
pullin_bottom(jive::gamma_node * gamma)
{
	/* collect immediate successors of the gamma node */
	std::unordered_set<jive::node*> workset;
	for (size_t n = 0; n < gamma->noutputs(); n++) {
		auto output = gamma->output(n);
		for (const auto & user : *output) {
			if (user->node() && user->node()->depth() == gamma->depth()+1)
				workset.insert(user->node());
		}
	}

	while (!workset.empty()) {
		auto node = *workset.begin();
		workset.erase(node);

		/* copy node into subregions */
		std::vector<std::vector<jive::output*>> outputs(node->noutputs());
		for (size_t r = 0; r < gamma->nsubregions(); r++) {
			/* collect operands */
			std::vector<jive::output*> operands;
			for (size_t i = 0; i < node->ninputs(); i++) {
				auto input = node->input(i);
				if (input->origin()->node() == gamma) {
					auto output = static_cast<jive::structural_output*>(input->origin());
					operands.push_back(gamma->subregion(r)->result(output->index())->origin());
				} else {
					auto ev = gamma->add_entryvar(input->origin());
					operands.push_back(ev->argument(r));
				}
			}

			auto copy = node->copy(gamma->subregion(r), operands);
			for (size_t o = 0; o < copy->noutputs(); o++)
				outputs[o].push_back(copy->output(o));
		}

		/* adjust outputs and update workset */
		for (size_t n = 0; n < node->noutputs(); n++) {
			auto output = node->output(n);
			for (const auto & user : *output)
				if (user->node()  && user->node()->depth() == node->depth()+1)
					workset.insert(user->node());

			auto xv = gamma->add_exitvar(outputs[n]);
			output->divert_users(xv);
		}
	}
}

static size_t
is_used_in_nsubregions(const jive::gamma_node * gamma, const jive::node * node)
{
	JLM_DEBUG_ASSERT(single_successor(node));

	/* collect all gamma inputs */
	std::unordered_set<const jive::gamma_input*> inputs;
	for (size_t n = 0; n < node->noutputs(); n++) {
		for (const auto & user : *(node->output(n))) {
			JLM_DEBUG_ASSERT(is_gamma_input(user));
			inputs.insert(static_cast<const jive::gamma_input*>(user));
		}
	}

	/* collect subregions where node is used */
	std::unordered_set<jive::region*> subregions;
	for (const auto & input : inputs) {
		for (const auto & argument : *input) {
			if (argument.nusers() != 0)
				subregions.insert(argument.region());
		}
	}

	return subregions.size();
}

void
pull(jive::gamma_node * gamma)
{
	/*
		We don't want to pull anything into empty gammas with two subregions,
		as they are translated to select instructions in the r2j phase.
	*/
	if (gamma->nsubregions() == 2 && empty(gamma))
		return;

	auto prednode = gamma->predicate()->origin()->node();

	/* FIXME: This is inefficient. We can do better. */
	auto ev = gamma->begin_entryvar();
	while (ev != gamma->end_entryvar()) {
		auto node = ev->origin()->node();
		if (!node || prednode == node || !single_successor(node)) {
			ev++; continue;
		}

		if (is_used_in_nsubregions(gamma, node) == 1) {
			/*
				FIXME: This function pulls in the node to ALL subregions and
				not just the one we care about.
			*/
			pullin_node(gamma, node);
			cleanup(gamma, node);
			ev = gamma->begin_entryvar();
		} else {
			ev++;
		}
	}
}

void
pull(jive::region * region)
{
	for (auto & node : jive::topdown_traverser(region)) {
		if (auto structnode = dynamic_cast<jive::structural_node*>(node)) {
			if (auto gamma = dynamic_cast<jive::gamma_node*>(node))
				pull(gamma);

			for (size_t n = 0; n < structnode->nsubregions(); n++)
				pull(structnode->subregion(n));
		}
	}
}

void
pull(jlm::rvsdg & rvsdg)
{
	pull(rvsdg.graph()->root());
}

}

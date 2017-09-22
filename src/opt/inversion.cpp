/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/opt/inversion.hpp>

#include <jive/vsdg/gamma.h>
#include <jive/vsdg/substitution.h>
#include <jive/vsdg/theta.h>
#include <jive/vsdg/traverser.h>

namespace jlm {

static jive::structural_node *
is_applicable(jive::structural_node * node)
{
	JLM_DEBUG_ASSERT(is_theta_node(node));
	jive::theta theta(node);

	auto matchnode = theta.predicate()->origin()->node();
	if (!jive::is_opnode<jive::ctl::match_op>(matchnode))
		return nullptr;

	if (matchnode->output(0)->nusers() != 2)
		return nullptr;

	jive::structural_node * gnode = nullptr;
	for (const auto & user : *matchnode->output(0)) {
		if (user == theta.predicate())
			continue;

		if (!is_gamma_node(user->node()))
			return nullptr;

		gnode = dynamic_cast<jive::structural_node*>(user->node());
	}
	JLM_DEBUG_ASSERT(gnode);

	for (const auto & lv : theta) {
		if (lv.result()->origin() == lv.argument())
			continue;

		if (lv.result()->origin()->node() == gnode)
			continue;

		return nullptr;
	}

	jive::gamma gamma(gnode);
	for (auto ev = gamma.begin_entryvar(); ev != gamma.end_entryvar(); ev++) {
		if (!dynamic_cast<const jive::argument*>(ev->input()->origin()))
			return nullptr;
	}

	return gnode;
}

static std::vector<std::vector<jive::node*>>
collect_condition_nodes(
	jive::structural_node * tnode,
	jive::structural_node * gnode)
{
	JLM_DEBUG_ASSERT(is_theta_node(tnode));
	JLM_DEBUG_ASSERT(is_gamma_node(gnode));
	JLM_DEBUG_ASSERT(gnode->region()->node() == tnode);

	std::vector<std::vector<jive::node*>> nodes;
	for (auto & node : tnode->subregion(0)->nodes) {
		if (&node == gnode)
			continue;

		if (node.depth() >= nodes.size())
			nodes.resize(node.depth()+1);
		nodes[node.depth()].push_back(&node);
	}

	return nodes;
}

static void
copy_condition_nodes(
	jive::region * target,
	jive::substitution_map & smap,
	const std::vector<std::vector<jive::node*>> & nodes)
{
	for (size_t n = 0; n < nodes.size(); n++) {
		for (const auto & node : nodes[n])
			node->copy(target, smap);
	}
}

static void
invert(jive::region * region);

static void
invert_theta(jive::structural_node * tnode)
{
	JLM_DEBUG_ASSERT(is_theta_node(tnode));
	jive::theta otheta(tnode);

	auto gnode = is_applicable(tnode);
	if (!gnode) {
		invert(tnode->subregion(0));
		return;
	}

	jive::gamma ogamma(gnode);

	/* copy condition nodes for new gamma node */
	jive::substitution_map smap;
	auto cnodes = collect_condition_nodes(tnode, gnode);
	for (const auto & olv : otheta)
		smap.insert(olv.argument(), olv.input()->origin());
	copy_condition_nodes(tnode->region(), smap, cnodes);

	jive::gamma_builder gb;
	gb.begin_gamma(smap.lookup(ogamma.predicate()->origin()));

	jive::substitution_map rmap[gb.nsubregions()];
	for (size_t r = 0; r < gb.nsubregions(); r++) {
		auto nsubregion = gb.subregion(r);

		jive::theta_builder tb;
		tb.begin_theta(nsubregion);

		/* add all loop variables to new gamma and theta */
		for (const auto & olv : otheta) {
			auto ev = gb.add_entryvar(olv.input()->origin());
			auto nlv = tb.add_loopvar(ev->argument(r));
			rmap[r].insert(olv.argument(), nlv->argument());
		}

		/* insert arguments of old gamma subregion into substitution map */
		auto osubregion = ogamma.subregion(r);
		for (size_t n = 0; n < osubregion->narguments(); n++) {
			auto argument = osubregion->argument(n);
			rmap[r].insert(argument, rmap[r].lookup(argument->input()->origin()));
		}

		/* copy condition nodes and old gamma subregion into new theta subregion */
		copy_condition_nodes(tb.subregion(), rmap[r], cnodes);
		auto predicate = rmap[r].lookup(ogamma.predicate()->origin());
		osubregion->copy(tb.subregion(), rmap[r], false, false);

		auto ntheta = tb.end_theta(predicate);

		/* redirect new loop variable results to right origin */
		JLM_DEBUG_ASSERT(otheta.nloopvars() == ntheta->nloopvars());
		for (auto olv = otheta.begin(), nlv = ntheta->begin(); olv != otheta.end(); olv++, nlv++) {
			auto origin = olv->result()->origin();
			if (origin->node() && origin->node() == ogamma.node()) {
				auto output = static_cast<jive::structural_output*>(origin);
				auto substitute = rmap[r].lookup(ogamma.subregion(r)->result(output->index())->origin());
				nlv->result()->divert_origin(substitute);
			}
			rmap[r].insert(origin, nlv->output());
		}
	}

	/* add exit variables to new gamma */
	for (const auto & olv : otheta) {
		std::vector<jive::output*> outputs;
		for (size_t r = 0; r < gb.nsubregions(); r++)
			outputs.push_back(rmap[r].lookup(olv.result()->origin()));
		auto ex = gb.add_exitvar(outputs);
		smap.insert(olv.output(), ex->output());
	}

	auto ngamma = gb.end_gamma();

	/* replace outputs */
	for (const auto & olv : otheta)
		olv.output()->replace(smap.lookup(olv.output()));

	for (size_t n = 0; n < ngamma->nsubregions(); n++)
		invert(ngamma->subregion(n));
}

static void
invert(jive::structural_node * node)
{
	if (is_theta_node(node)) {
		invert_theta(node);
		return;
	}

	for (size_t n = 0; n < node->nsubregions(); n++)
		invert(node->subregion(n));
}

static void
invert(jive::region * region)
{
	for (auto & node : jive::topdown_traverser(region)) {
		if (auto structnode = dynamic_cast<jive::structural_node*>(node))
			invert(structnode);
	}
}

void
invert(jive::graph & graph)
{
	auto root = graph.root();
	invert(root);
}

}

/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/opt/inversion.hpp>
#include <jlm/opt/pull.hpp>

#include <jive/vsdg/gamma.h>
#include <jive/vsdg/substitution.h>
#include <jive/vsdg/theta.h>
#include <jive/vsdg/traverser.h>

namespace jlm {

static jive::structural_node *
is_applicable(const jive::theta & theta)
{
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

	return gnode;
}

static void
pullin(jive::gamma & gamma, jive::theta & theta)
{
	pullin_bottom(gamma);
	for (const auto & lv : theta)	{
		if (lv.result()->origin()->node() != gamma.node()) {
			auto ev = gamma.add_entryvar(lv.result()->origin());
			JLM_DEBUG_ASSERT(ev->narguments() == 2);
			auto xv = gamma.add_exitvar({ev->argument(0), ev->argument(1)});
			lv.result()->divert_origin(xv->output());
		}
	}
	pullin_top(gamma);
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

static jive::argument *
to_argument(jive::output * output)
{
	return dynamic_cast<jive::argument*>(output);
}

static jive::structural_output *
to_structural_output(jive::output * output)
{
	return dynamic_cast<jive::structural_output*>(output);
}

static void
invert(jive::theta & otheta)
{
	auto gnode = is_applicable(otheta);
	if (!gnode) return;

	jive::gamma ogamma(gnode);
	pullin(ogamma, otheta);

	/* copy condition nodes for new gamma node */
	jive::substitution_map smap;
	auto cnodes = collect_condition_nodes(otheta.node(), gnode);
	for (const auto & olv : otheta)
		smap.insert(olv.argument(), olv.input()->origin());
	copy_condition_nodes(otheta.node()->region(), smap, cnodes);

	jive::gamma_builder gb;
	gb.begin_gamma(smap.lookup(ogamma.predicate()->origin()));

	/* handle subregion 0 */
	jive::substitution_map r0map;
	{
		/* setup substitution map for exit region copying */
		auto osubregion0 = ogamma.subregion(0);
		for (auto oev = ogamma.begin_entryvar(); oev != ogamma.end_entryvar(); oev++) {
			if (auto argument = to_argument(oev->input()->origin())) {
				auto nev = gb.add_entryvar(argument->input()->origin());
				r0map.insert(oev->argument(0), nev->argument(0));
			} else {
				auto substitute = smap.lookup(oev->input()->origin());
				auto nev = gb.add_entryvar(substitute);
				r0map.insert(oev->argument(0), nev->argument(0));
			}
		}

		/* copy exit region */
		osubregion0->copy(gb.subregion(0), r0map, false, false);

		/* update substitution map for insertion of exit variables */
		for (const auto & olv : otheta) {
			auto output = to_structural_output(olv.result()->origin());
			auto substitute = r0map.lookup(osubregion0->result(output->index())->origin());
			r0map.insert(olv.result()->origin(), substitute);
		}
	}

	/* handle subregion 1 */
	jive::substitution_map r1map;
	{
		jive::theta_builder tb;
		tb.begin_theta(gb.subregion(1));

		/* add loop variables to new theta node and setup substitution map */
		auto osubregion0 = ogamma.subregion(0);
		auto osubregion1 = ogamma.subregion(1);
		std::unordered_map<jive::input*, jive::loopvar> nlvs;
		for (const auto & olv : otheta) {
			auto ev = gb.add_entryvar(olv.input()->origin());
			auto nlv = tb.add_loopvar(ev->argument(1));
			r1map.insert(olv.argument(), nlv->argument());
			nlvs[olv.input()] = *nlv;
		}
		for (auto oev = ogamma.begin_entryvar(); oev != ogamma.end_entryvar(); oev++) {
			if (auto argument = to_argument(oev->input()->origin())) {
				r1map.insert(oev->argument(1), nlvs[argument->input()].argument());
			} else {
				auto ev = gb.add_entryvar(smap.lookup(oev->input()->origin()));
				auto nlv = tb.add_loopvar(ev->argument(1));
				r1map.insert(oev->argument(1), nlv->argument());
				nlvs[oev->input()] = *nlv;
			}
		}

		/* copy repetition region  */
		osubregion1->copy(tb.subregion(), r1map, false, false);

		/* adjust values in substitution map for condition node copying */
		for (const auto & olv : otheta) {
			auto output = to_structural_output(olv.result()->origin());
			auto substitute = r1map.lookup(osubregion1->result(output->index())->origin());
			r1map.insert(olv.argument(), substitute);
		}

		/* copy condition nodes */
		copy_condition_nodes(tb.subregion(), r1map, cnodes);
		auto predicate = r1map.lookup(ogamma.predicate()->origin());

		/* redirect results of loop variables and adjust substitution map for exit region copying */
		for (const auto & olv : otheta) {
			auto output = to_structural_output(olv.result()->origin());
			auto substitute = r1map.lookup(osubregion1->result(output->index())->origin());
			nlvs[olv.input()].result()->divert_origin(substitute);
			r1map.insert(olv.result()->origin(), nlvs[olv.input()].output());
		}
		for (auto oev = ogamma.begin_entryvar(); oev != ogamma.end_entryvar(); oev++) {
			if (auto argument = to_argument(oev->input()->origin())) {
				r1map.insert(oev->argument(0), nlvs[argument->input()].output());
			} else {
				auto substitute = r1map.lookup(oev->input()->origin());
				nlvs[oev->input()].result()->divert_origin(substitute);
				r1map.insert(oev->argument(0), nlvs[oev->input()].output());
			}
		}

		tb.end_theta(predicate);

		/* copy exit region */
		osubregion0->copy(gb.subregion(1), r1map, false, false);

		/* adjust values in substitution map for exit variable creation */
		for (const auto & olv : otheta) {
			auto output = to_structural_output(olv.result()->origin());
			auto substitute = r1map.lookup(osubregion0->result(output->index())->origin());
			r1map.insert(olv.result()->origin(), substitute);
		}

	}

	/* add exit variables to new gamma */
	for (const auto & olv : otheta) {
		auto o0 = r0map.lookup(olv.result()->origin());
		auto o1 = r1map.lookup(olv.result()->origin());
		auto ex = gb.add_exitvar({o0, o1});
		smap.insert(olv.output(), ex->output());
	}

	auto ngamma = gb.end_gamma();

	/* replace outputs */
	for (const auto & olv : otheta)
		olv.output()->replace(smap.lookup(olv.output()));
}

static void
invert(jive::region * region)
{
	for (auto & node : jive::topdown_traverser(region)) {
		if (auto structnode = dynamic_cast<jive::structural_node*>(node)) {
			for (size_t r = 0; r < structnode->nsubregions(); r++)
				invert(structnode->subregion(r));

			if (is_theta_node(structnode)) {
				jive::theta theta(structnode);
				invert(theta);
			}
		}
	}
}

void
invert(jive::graph & graph)
{
	auto root = graph.root();
	invert(root);
}

}

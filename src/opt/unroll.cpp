/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/types/bitstring/arithmetic.h>
#include <jive/types/bitstring/comparison.h>
#include <jive/types/bitstring/constant.h>
#include <jive/vsdg/structural_node.h>
#include <jive/vsdg/substitution.h>
#include <jive/vsdg/theta.h>
#include <jive/vsdg/traverser.h>

#include <jlm/common.hpp>
#include <jlm/opt/unroll.hpp>

#include <deque>

namespace jlm {

static std::vector<std::vector<jive::node*>>
compute_slice(jive::input * input)
{
	if (!input->origin()->node())
		return {};

	std::vector<std::vector<jive::node*>> nodes;
	std::deque<jive::node*> worklist({input->origin()->node()});
	while (!worklist.empty()) {
		auto node = worklist.front();
		worklist.pop_front();

		if (node->depth() >= nodes.size())
			nodes.resize(node->depth()+1);
		nodes[node->depth()].push_back(node);

		for (size_t n = 0; n < node->ninputs(); n++) {
			auto input = node->input(n);
			if (!input->origin()->node())
				continue;

			worklist.push_back(input->origin()->node());
		}
	}

	return nodes;
}

static void
copy_nodes(
	jive::region * target,
	jive::substitution_map & smap,
	const std::vector<std::vector<jive::node*>> & nodes)
{
	for (size_t n = 0; n < nodes.size(); n++) {
		for (const auto & node : nodes[n])
			node->copy(target, smap);
	}
}

static bool
has_side_effects(const jive::node * node)
{
	for (size_t n = 0; n < node->noutputs(); n++) {
		if (dynamic_cast<const jive::state::type*>(&node->output(n)->type()))
			return true;
	}

	return false;
}

static bool
contains_theta(const jive::region * region)
{
	for (const auto & node : *region) {
		if (jive::is_theta_node(&node))
			return true;

		if (auto structnode = dynamic_cast<const jive::structural_node*>(&node)) {
			for (size_t r = 0; r < structnode->nsubregions(); r++)
				contains_theta(structnode->subregion(r));
		}
	}

	return false;
}

static bool
is_applicable(const jive::theta & theta)
{
	auto matchnode = theta.predicate()->origin()->node();
	if (!jive::is_opnode<jive::ctl::match_op>(matchnode))
		return false;

	auto cmpnode = matchnode->input(0)->origin()->node();
	if (!jive::is_opnode<jive::bits::compare_op>(cmpnode))
		return false;

	if (jive::is_opnode<jive::bits::eq_op>(cmpnode)
	|| jive::is_opnode<jive::bits::ne_op>(cmpnode))
		return false;

	auto cnodes = compute_slice(matchnode->input(0));
	for (const auto & nodes : cnodes) {
		for (const auto & node : nodes) {
			if (has_side_effects(node))
				return false;
		}
	}

	if (contains_theta(theta.subregion()))
		return false;

	return true;
}

struct thetainfo {
	inline
	thetainfo(
		size_t nbits_,
		jive::output * min_,
		jive::output * max_,
		bool is_signed_)
	: nbits(nbits_)
	, min(min_)
	, max(max_)
	, is_signed(is_signed_)
	{}

	size_t nbits;
	jive::output * min;
	jive::output * max;
	bool is_signed;
};

static bool
is_signed(const jive::bits::compare_op & op)
{
	return dynamic_cast<const jive::bits::sge_op*>(&op)
	    || dynamic_cast<const jive::bits::sgt_op*>(&op)
	    || dynamic_cast<const jive::bits::sle_op*>(&op)
	    || dynamic_cast<const jive::bits::slt_op*>(&op);
}

static bool
is_greaterop(const jive::bits::compare_op & op)
{
	return dynamic_cast<const jive::bits::uge_op*>(&op)
	    || dynamic_cast<const jive::bits::ugt_op*>(&op)
	    || dynamic_cast<const jive::bits::sge_op*>(&op)
	    || dynamic_cast<const jive::bits::sgt_op*>(&op);
}

static struct thetainfo
compute_thetainfo(const jive::theta & theta)
{
	auto matchnode = theta.predicate()->origin()->node();
	JLM_DEBUG_ASSERT(jive::is_opnode<jive::ctl::match_op>(matchnode));
	auto cmpnode = matchnode->input(0)->origin()->node();
	JLM_DEBUG_ASSERT(jive::is_opnode<jive::bits::compare_op>(cmpnode));
	auto cmpop = static_cast<const jive::bits::compare_op*>(&cmpnode->operation());
	auto nbits = static_cast<const jive::bits::type*>(&cmpop->argument(0).type())->nbits();

	bool issigned = is_signed(*cmpop);
	auto min = is_greaterop(*cmpop) ? cmpnode->input(1) : cmpnode->input(0);
	auto max = is_greaterop(*cmpop) ? cmpnode->input(0) : cmpnode->input(1);

	return {nbits, min->origin(), max->origin(), issigned};
}

static std::unique_ptr<jive::theta>
create_unrolled_theta(
	jive::region * target,
	jive::substitution_map & smap,
	const jive::theta & otheta,
	size_t factor)
{
	jive::theta_builder tb;
	tb.begin_theta(target);

	for (const auto & olv : otheta) {
		auto nlv = tb.add_loopvar(smap.lookup(olv.input()->origin()));
		smap.insert(olv.argument(), nlv->argument());
	}

	while (factor--) {
		otheta.subregion()->copy(tb.subregion(), smap, false, false);
		for (const auto & olv : otheta) {
			auto v = smap.lookup(olv.result()->origin());
			smap.insert(olv.argument(), v);
		}
	}

	for (auto olv = otheta.begin(), nlv = tb.begin(); olv != otheta.end(); olv++, nlv++) {
		auto origin = smap.lookup(olv->result()->origin());
		nlv->result()->divert_origin(origin);
		smap.insert(olv->output(), nlv->output());
	}

	return tb.end_theta(smap.lookup(otheta.predicate()->origin()));
}

static void
unroll(jive::theta & theta, size_t factor)
{
	if (!is_applicable(theta))
		return;

	jive::substitution_map smap;
	auto info = compute_thetainfo(theta);
	auto cnodes = compute_slice(theta.predicate());

	/* handle gamma with unrolled loop */
	{
		for (const auto & olv : theta)
			smap.insert(olv.argument(), olv.input()->origin());
		copy_nodes(theta.node()->region(), smap, cnodes);

		auto uf = jive::create_bitconstant(theta.node()->region(), info.nbits, factor);
		auto sub = jive::bits::create_sub(info.nbits, smap.lookup(info.max), smap.lookup(info.min));
		auto cmp = jive::bits::create_uge(info.nbits, sub, uf);
		auto pred = jive::ctl::match(1, {{1, 0}}, 1, 2, cmp);

		jive::gamma_builder gb;
		gb.begin_gamma(pred);

		jive::substitution_map rmap[2];
		for (const auto & olv : theta) {
			auto ev = gb.add_entryvar(olv.input()->origin());
			rmap[0].insert(olv.input()->origin(), ev->argument(0));
			rmap[1].insert(olv.output(), ev->argument(1));
		}

		auto utheta = create_unrolled_theta(gb.subregion(0), rmap[0], theta, factor);

		for (const auto & olv : theta) {
			auto output = olv.output();
			auto xv = gb.add_exitvar({rmap[0].lookup(output), rmap[1].lookup(output)});
			smap.insert(olv.output(), xv->output());
		}

		auto gamma = gb.end_gamma();
	}

	/* handle gamma for leftover iterations */
	{
		for (const auto & olv : theta)
			smap.insert(olv.argument(), smap.lookup(olv.output()));
		copy_nodes(theta.node()->region(), smap, cnodes);

		auto zero = jive::create_bitconstant(theta.node()->region(), info.nbits, 0);
		auto sub = jive::bits::create_sub(info.nbits, smap.lookup(info.max), smap.lookup(info.min));
		auto cmp = jive::bits::create_ne(info.nbits, sub, zero);
		auto pred = jive::ctl::match(1, {{1, 0}}, 1, 2, cmp);

		jive::gamma_builder gb;
		gb.begin_gamma(pred);

		jive::substitution_map rmap[2];
		for (const auto & olv : theta) {
			auto ev = gb.add_entryvar(smap.lookup(olv.output()));
			rmap[0].insert(olv.input()->origin(), ev->argument(0));
			rmap[1].insert(olv.output(), ev->argument(1));
		}

		theta.node()->copy(gb.subregion(0), rmap[0]);

		for (const auto & olv : theta) {
			auto output = olv.output();
			auto xv = gb.add_exitvar({rmap[0].lookup(output), rmap[1].lookup(output)});
			smap.insert(olv.output(), xv->output());
		}

		auto gamma = gb.end_gamma();
	}

	for (const auto & olv : theta)
		olv.output()->replace(smap.lookup(olv.output()));
}

static void
unroll(jive::region * region, size_t factor)
{
	for (auto & node : jive::topdown_traverser(region)) {
		if (auto structnode = dynamic_cast<jive::structural_node*>(node)) {
			for (size_t n = 0; n < structnode->nsubregions(); n++)
				unroll(structnode->subregion(n), factor);

			if (is_theta_node(node)) {
				jive::theta theta(structnode);
				unroll(theta, factor);
			}
		}
	}
}

void
unroll(jive::graph & rvsdg, size_t factor)
{
	if (factor == 0)
		return;

	unroll(rvsdg.root(), factor);
}

}

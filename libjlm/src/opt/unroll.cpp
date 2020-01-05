/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/arch/addresstype.h>
#include <jive/types/bitstring/arithmetic.h>
#include <jive/types/bitstring/comparison.h>
#include <jive/types/bitstring/constant.h>
#include <jive/rvsdg/binary.h>
#include <jive/rvsdg/gamma.h>
#include <jive/rvsdg/structural-node.h>
#include <jive/rvsdg/substitution.h>
#include <jive/rvsdg/theta.h>
#include <jive/rvsdg/traverser.h>

#include <jlm/common.hpp>
#include <jlm/ir/rvsdg.hpp>
#include <jlm/opt/unroll.hpp>

namespace jlm {

/* helper functions */

static bool
contains_theta(const jive::region * region)
{
	for (const auto & node : *region) {
		if (jive::is<jive::theta_op>(&node))
			return true;

		if (auto structnode = dynamic_cast<const jive::structural_node*>(&node)) {
			for (size_t r = 0; r < structnode->nsubregions(); r++)
				if (contains_theta(structnode->subregion(r)))
					return true;
		}
	}

	return false;
}

static bool
is_eqcmp(const jive::operation & op)
{
	return dynamic_cast<const jive::bituge_op*>(&op)
	    || dynamic_cast<const jive::bitsge_op*>(&op)
	    || dynamic_cast<const jive::bitule_op*>(&op)
	    || dynamic_cast<const jive::bitsle_op*>(&op);
}

/* unrollinfo methods */

static bool
is_theta_invariant(const jive::output * output)
{
	JLM_DEBUG_ASSERT(jive::is<jive::theta_op>(output->region()->node()));

	if (jive::is<jive::bitconstant_op>(output->node()))
		return true;

	auto argument = dynamic_cast<const jive::argument*>(output);
	if (!argument)
		return false;

	return is_invariant(static_cast<const jive::theta_input*>(argument->input()));
}

static jive::argument *
push_from_theta(jive::output * output)
{
	auto argument = dynamic_cast<jive::argument*>(output);
	if (argument) return argument;

	JLM_DEBUG_ASSERT(jive::is<jive::bitconstant_op>(output->node()));
	JLM_DEBUG_ASSERT(jive::is<jive::theta_op>(output->node()->region()->node()));
	auto theta = static_cast<jive::theta_node*>(output->node()->region()->node());

	auto node = output->node()->copy(theta->region(), {});
	auto lv = theta->add_loopvar(node->output(0));
	output->divert_users(lv->argument());

	return lv->argument();
}

static bool
is_idv(jive::input * input)
{
	using namespace jive;

	auto node = input->node();
	JLM_DEBUG_ASSERT(is<bitadd_op>(node) || is<bitsub_op>(node));

	auto a = dynamic_cast<argument*>(input->origin());
	if (!a) return false;

	auto tinput = static_cast<const theta_input*>(a->input());
	return tinput->result()->origin()->node() == node;
}

std::unique_ptr<jive::bitvalue_repr>
unrollinfo::niterations() const noexcept
{
	if (!is_known() || step_value() == 0)
		return nullptr;

	auto start = is_additive() ? *init_value() : *end_value();
	auto step = is_additive() ? *step_value() : step_value()->neg();
	auto end = is_additive() ? *end_value() : *init_value();

	if (is_eqcmp(cmpnode()->operation()))
		end = end.add({nbits(), 1});

	auto range = end.sub(start);
	if (range.is_negative())
		return nullptr;

	if (range.umod(step) != 0)
		return nullptr;

	return std::make_unique<jive::bitvalue_repr>(range.udiv(step));
}

std::unique_ptr<unrollinfo>
unrollinfo::create(jive::theta_node * theta)
{
	using namespace jive;

	auto matchnode = theta->predicate()->origin()->node();
	if (!is<match_op>(matchnode))
		return nullptr;

	auto cmpnode = matchnode->input(0)->origin()->node();
	if (!is<bitcompare_op>(cmpnode))
		return nullptr;

	auto o0 = cmpnode->input(0)->origin();
	auto o1 = cmpnode->input(1)->origin();
	auto end = is_theta_invariant(o0) ? o0 : (is_theta_invariant(o1) ? o1 : nullptr);
	if (!end) return nullptr;

	auto armnode = (end == o0 ? o1 : o0)->node();
	if (!is<bitadd_op>(armnode) && !is<bitsub_op>(armnode))
		return nullptr;
	if (armnode->ninputs() != 2)
		return nullptr;

	auto i0 = armnode->input(0);
	auto i1 = armnode->input(1);
	if (!is_idv(i0) && !is_idv(i1))
		return nullptr;

	auto idv = static_cast<argument*>(is_idv(i0) ? i0->origin() : i1->origin());

	auto step = idv == i0->origin() ? i1->origin() : i0->origin();
	if (!is_theta_invariant(step))
		return nullptr;

	auto endarg = push_from_theta(end);
	auto steparg = push_from_theta(step);
	return std::unique_ptr<unrollinfo>(new unrollinfo(cmpnode, armnode, idv, steparg, endarg));
}

/* loop unrolling */

static std::unique_ptr<unrollinfo>
is_unrollable(jive::theta_node * theta)
{
	if (contains_theta(theta->subregion()))
		return nullptr;

	return unrollinfo::create(theta);
}

static void
unroll_body(
	const jive::theta_node * theta,
	jive::region * target,
	jive::substitution_map & smap,
	size_t ntimes)
{
	for (size_t n = 0; n < ntimes-1; n++) {
		theta->subregion()->copy(target, smap, false, false);
		jive::substitution_map tmap;
		for (const auto & olv : *theta)
			tmap.insert(olv->argument(), smap.lookup(olv->result()->origin()));
		smap = tmap;
	}
	theta->subregion()->copy(target, smap, false, false);
}

static void
unroll_known_theta(const unrollinfo & ui, size_t factor)
{
	JLM_DEBUG_ASSERT(ui.is_known() && ui.niterations());
	auto niterations = ui.niterations();
	auto cmpnode = ui.cmpnode();
	auto otheta = ui.theta();
	auto nbits = ui.nbits();

	JLM_DEBUG_ASSERT(niterations != 0);
	if (niterations->ule({nbits, (int64_t)factor}) == '1') {
		/*
			Completely unroll the loop body and remove the theta node, as
			the number of iterations is smaller than the unroll factor.
		*/
		jive::substitution_map smap;
		for (const auto & olv : *otheta)
			smap.insert(olv->argument(), olv->input()->origin());

		unroll_body(otheta, otheta->region(), smap, niterations->to_uint());

		for (const auto & olv : *otheta)
			olv->divert_users(smap.lookup(olv->result()->origin()));
		return remove(otheta);
	}

	/*
		Unroll theta node by given factor.
	*/
	JLM_DEBUG_ASSERT(niterations->ugt({nbits, (int64_t)factor}) == '1');

	jive::substitution_map smap;
	auto ntheta = jive::theta_node::create(otheta->region());
	for (const auto & olv : *otheta) {
		auto nlv = ntheta->add_loopvar(olv->input()->origin());
		smap.insert(olv->argument(), nlv->argument());
	}

	unroll_body(otheta, ntheta->subregion(), smap, factor);
	ntheta->set_predicate(smap.lookup(otheta->predicate()->origin()));

	for (auto olv = otheta->begin(), nlv = ntheta->begin(); olv != otheta->end(); olv++, nlv++) {
		auto origin = smap.lookup((*olv)->result()->origin());
		(*nlv)->result()->divert_to(origin);
		smap.insert(*olv, *nlv);
	}

	auto remainder = niterations->umod({nbits, (int64_t)factor});
	if (remainder == 0) {
		/*
			We only need to redirect the users of the outputs of the old theta node
			to the outputs of the new theta node, as there are no residual iterations.
		*/
		for (const auto & olv : *otheta)
			olv->divert_users(smap.lookup(olv));
		return remove(otheta);
	}

	/*
		We have residual iterations. Adjust the end value of the unrolled loop to a
		multiple of the step value, and add an epilogue after the unrolled loop that
		computes the residual iterations.
	*/
	auto cmp = smap.lookup(cmpnode->output(0))->node();
	auto input = cmp->input(0)->origin() == smap.lookup(ui.end()) ? cmp->input(0) : cmp->input(1);
	JLM_DEBUG_ASSERT(input->origin() == smap.lookup(ui.end()));

	auto sv = ui.is_additive() ? *ui.step_value() : ui.step_value()->neg();
	/* FIXME: What is if we need a high mul operation? */
	auto end = remainder.mul(sv);
	auto ev = ui.is_additive() ? ui.end_value()->sub(end) : ui.end_value()->add(end);

	auto c = jive::create_bitconstant(ntheta->subregion(), ev);
	input->divert_to(c);

	if (remainder == 1) {
		/*
				There is only one loop iteration remaining. Just copy the loop body
				after the unrolled loop and remove the old loop.
		*/
		jive::substitution_map rmap;
		for (const auto & olv : *otheta)
			rmap.insert(olv->argument(), smap.lookup(olv));

		otheta->subregion()->copy(ntheta->region(), rmap, false, false);

		for (const auto & olv : *otheta)
			olv->divert_users(rmap.lookup(olv->result()->origin()));

		return remove(otheta);
	}

	/*
		Add the old theta node as epilogue after the unrolled loop by simply
		redirecting the inputs of the old theta to the outputs of the unrolled
		theta.
	*/
	for (const auto & olv : *otheta)
		olv->input()->divert_to(smap.lookup(olv));
}

static jive::output *
create_unrolled_gamma_predicate(
	const unrollinfo & ui,
	size_t factor)
{
	auto region = ui.theta()->region();
	auto nbits = ui.nbits();
	auto step = ui.step()->input()->origin();
	auto end = ui.end()->input()->origin();

	auto uf = jive::create_bitconstant(region, nbits, factor);
	auto mul = jive::bitmul_op::create(nbits, step, uf);
	auto arm = jive::simple_node::create_normalized(region, ui.armoperation(), {ui.init(), mul})[0];
	/* FIXME: order of operands */
	auto cmp = jive::simple_node::create_normalized(region, ui.cmpoperation(), {arm, end})[0];
	auto pred = jive::match(1, {{1, 1}}, 0, 2, cmp);

	return pred;
}

static jive::output *
create_unrolled_theta_predicate(
	jive::region * target,
	const jive::substitution_map & smap,
	const unrollinfo & ui,
	size_t factor)
{
	using namespace jive;

	auto region = smap.lookup(ui.cmpnode()->output(0))->region();
	auto cmpnode = smap.lookup(ui.cmpnode()->output(0))->node();
	auto step = smap.lookup(ui.step());
	auto end = smap.lookup(ui.end());
	auto nbits = ui.nbits();

	auto i0 = cmpnode->input(0);
	auto i1 = cmpnode->input(1);
	auto iend = i0->origin() == end ? i0 : i1;
	auto idv = i0->origin() == end ? i1 : i0;

	auto uf = create_bitconstant(region, nbits, factor);
	auto mul = bitmul_op::create(nbits, step, uf);
	auto arm = simple_node::create_normalized(region, ui.armoperation(), {idv->origin(), mul})[0];
	/* FIXME: order of operands */
	auto cmp = simple_node::create_normalized(region, ui.cmpoperation(), {arm, iend->origin()})[0];
	auto pred = match(1, {{1, 1}}, 0, 2, cmp);

	return pred;
}

static jive::output *
create_residual_gamma_predicate(
	const jive::substitution_map & smap,
	const unrollinfo & ui)
{
	auto region = ui.theta()->region();
	auto idv = smap.lookup(ui.theta()->output(ui.idv()->input()->index()));
	auto end = ui.end()->input()->origin();

	/* FIXME: order of operands */
	auto cmp = jive::simple_node::create_normalized(region, ui.cmpoperation(), {idv, end})[0];
	auto pred = jive::match(1, {{1, 1}}, 0, 2, cmp);

	return pred;
}

static void
unroll_unknown_theta(const unrollinfo & ui, size_t factor)
{
	auto otheta = ui.theta();

	/* handle gamma with unrolled loop */
	jive::substitution_map smap;
	{
		auto pred = create_unrolled_gamma_predicate(ui, factor);
		auto ngamma = jive::gamma_node::create(pred, 2);
		auto ntheta = jive::theta_node::create(ngamma->subregion(1));

		jive::substitution_map rmap[2];
		for (const auto & olv : *otheta) {
			auto ev = ngamma->add_entryvar(olv->input()->origin());
			auto nlv = ntheta->add_loopvar(ev->argument(1));
			rmap[0].insert(olv, ev->argument(0));
			rmap[1].insert(olv->argument(), nlv->argument());
		}

		unroll_body(otheta, ntheta->subregion(), rmap[1], factor);
		pred = create_unrolled_theta_predicate(ntheta->subregion(), rmap[1], ui, factor);
		ntheta->set_predicate(pred);

		for (auto olv = otheta->begin(), nlv = ntheta->begin(); olv != otheta->end(); olv++, nlv++) {
			auto origin = rmap[1].lookup((*olv)->result()->origin());
			(*nlv)->result()->divert_to(origin);
			rmap[1].insert(*olv, *nlv);
		}

		for (const auto & olv : *otheta) {
			auto xv = ngamma->add_exitvar({rmap[0].lookup(olv), rmap[1].lookup(olv)});
			smap.insert(olv, xv);
		}
	}

	/* handle gamma for residual iterations */
	{
		auto pred = create_residual_gamma_predicate(smap, ui);
		auto ngamma = jive::gamma_node::create(pred, 2);
		auto ntheta = jive::theta_node::create(ngamma->subregion(1));

		jive::substitution_map rmap[2];
		for (const auto & olv : *otheta) {
			auto ev = ngamma->add_entryvar(smap.lookup(olv));
			auto nlv = ntheta->add_loopvar(ev->argument(1));
			rmap[0].insert(olv, ev->argument(0));
			rmap[1].insert(olv->argument(), nlv->argument());
		}

		otheta->subregion()->copy(ntheta->subregion(), rmap[1], false, false);
		ntheta->set_predicate(rmap[1].lookup(otheta->predicate()->origin()));

		for (auto olv = otheta->begin(), nlv = ntheta->begin(); olv != otheta->end(); olv++, nlv++) {
			auto origin = rmap[1].lookup((*olv)->result()->origin());
			(*nlv)->result()->divert_to(origin);
			auto xv = ngamma->add_exitvar({rmap[0].lookup(*olv), *nlv});
			smap.insert(*olv, xv);
		}
	}

	for (const auto & olv : *otheta)
		olv->divert_users(smap.lookup(olv));
	remove(otheta);
}

void
unroll(jive::theta_node * otheta, size_t factor)
{
	if (factor < 2)
		return;

	auto ui = is_unrollable(otheta);
	if (!ui) return;

	auto nf = otheta->graph()->node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	if (ui->is_known() && ui->niterations())
		unroll_known_theta(*ui, factor);
	else
		unroll_unknown_theta(*ui, factor);

	nf->set_mutable(true);
}

static void
unroll(jive::region * region, size_t factor)
{
	for (auto & node : jive::topdown_traverser(region)) {
		if (auto structnode = dynamic_cast<jive::structural_node*>(node)) {
			for (size_t n = 0; n < structnode->nsubregions(); n++)
				unroll(structnode->subregion(n), factor);

			if (auto theta = dynamic_cast<jive::theta_node*>(node))
				unroll(theta, factor);
		}
	}
}

void
unroll(jlm::rvsdg & rvsdg, size_t factor)
{
	if (factor < 2)
		return;

	unroll(rvsdg.graph()->root(), factor);
}

}

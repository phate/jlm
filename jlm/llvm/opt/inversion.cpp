/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/inversion.hpp>
#include <jlm/llvm/opt/pull.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm {

class ivtstat final : public Statistics {
public:
	virtual
	~ivtstat()
	{}

	ivtstat()
	: Statistics(Statistics::Id::ThetaGammaInversion)
  , nnodes_before_(0), nnodes_after_(0)
	, ninputs_before_(0), ninputs_after_(0)
	{}

	void
	start(const jive::graph & graph) noexcept
	{
		nnodes_before_ = jive::nnodes(graph.root());
		ninputs_before_ = jive::ninputs(graph.root());
		timer_.start();
	}

	void
	end(const jive::graph & graph) noexcept
	{
		nnodes_after_ = jive::nnodes(graph.root());
		ninputs_after_ = jive::ninputs(graph.root());
		timer_.stop();
	}

	virtual std::string
	ToString() const override
	{
		return strfmt("IVT ",
			nnodes_before_, " ", nnodes_after_, " ",
			ninputs_before_, " ", ninputs_after_, " ",
			timer_.ns()
		);
	}

  static std::unique_ptr<ivtstat>
  Create()
  {
    return std::make_unique<ivtstat>();
  }

private:
	size_t nnodes_before_, nnodes_after_;
	size_t ninputs_before_, ninputs_after_;
	jlm::timer timer_;
};

static jive::gamma_node *
is_applicable(const jive::theta_node * theta)
{
	auto matchnode = jive::node_output::node(theta->predicate()->origin());
	if (!jive::is<jive::match_op>(matchnode))
		return nullptr;

	if (matchnode->output(0)->nusers() != 2)
		return nullptr;

	jive::gamma_node * gnode = nullptr;
	for (const auto & user : *matchnode->output(0)) {
		if (user == theta->predicate())
			continue;

		if (!jive::is<jive::gamma_op>(input_node(user)))
			return nullptr;

		gnode = dynamic_cast<jive::gamma_node*>(input_node(user));
	}

	return gnode;
}

static void
pullin(jive::gamma_node * gamma, jive::theta_node * theta)
{
	pullin_bottom(gamma);
	for (const auto & lv : *theta)	{
		if (jive::node_output::node(lv->result()->origin()) != gamma) {
			auto ev = gamma->add_entryvar(lv->result()->origin());
			JLM_ASSERT(ev->narguments() == 2);
			auto xv = gamma->add_exitvar({ev->argument(0), ev->argument(1)});
			lv->result()->divert_to(xv);
		}
	}
	pullin_top(gamma);
}

static std::vector<std::vector<jive::node*>>
collect_condition_nodes(
	jive::structural_node * tnode,
	jive::structural_node * gnode)
{
	JLM_ASSERT(jive::is<jive::theta_op>(tnode));
	JLM_ASSERT(jive::is<jive::gamma_op>(gnode));
	JLM_ASSERT(gnode->region()->node() == tnode);

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
invert(jive::theta_node * otheta)
{
	auto ogamma = is_applicable(otheta);
	if (!ogamma) return;

	pullin(ogamma, otheta);

	/* copy condition nodes for new gamma node */
	jive::substitution_map smap;
	auto cnodes = collect_condition_nodes(otheta, ogamma);
	for (const auto & olv : *otheta)
		smap.insert(olv->argument(), olv->input()->origin());
	copy_condition_nodes(otheta->region(), smap, cnodes);

	auto ngamma = jive::gamma_node::create(smap.lookup(ogamma->predicate()->origin()),
		ogamma->nsubregions());

	/* handle subregion 0 */
	jive::substitution_map r0map;
	{
		/* setup substitution map for exit region copying */
		auto osubregion0 = ogamma->subregion(0);
		for (auto oev = ogamma->begin_entryvar(); oev != ogamma->end_entryvar(); oev++) {
			if (auto argument = to_argument(oev->origin())) {
				auto nev = ngamma->add_entryvar(argument->input()->origin());
				r0map.insert(oev->argument(0), nev->argument(0));
			} else {
				auto substitute = smap.lookup(oev->origin());
				auto nev = ngamma->add_entryvar(substitute);
				r0map.insert(oev->argument(0), nev->argument(0));
			}
		}

		/* copy exit region */
		osubregion0->copy(ngamma->subregion(0), r0map, false, false);

		/* update substitution map for insertion of exit variables */
		for (const auto & olv : *otheta) {
			auto output = to_structural_output(olv->result()->origin());
			auto substitute = r0map.lookup(osubregion0->result(output->index())->origin());
			r0map.insert(olv->result()->origin(), substitute);
		}
	}

	/* handle subregion 1 */
	jive::substitution_map r1map;
	{
		auto ntheta = jive::theta_node::create(ngamma->subregion(1));

		/* add loop variables to new theta node and setup substitution map */
		auto osubregion0 = ogamma->subregion(0);
		auto osubregion1 = ogamma->subregion(1);
		std::unordered_map<jive::input*, jive::theta_output*> nlvs;
		for (const auto & olv : *otheta) {
			auto ev = ngamma->add_entryvar(olv->input()->origin());
			auto nlv = ntheta->add_loopvar(ev->argument(1));
			r1map.insert(olv->argument(), nlv->argument());
			nlvs[olv->input()] = nlv;
		}
		for (size_t n = 1; n < ogamma->ninputs(); n++) {
			auto oev = static_cast<jive::gamma_input*>(ogamma->input(n));
			if (auto argument = to_argument(oev->origin())) {
				r1map.insert(oev->argument(1), nlvs[argument->input()]->argument());
			} else {
				auto ev = ngamma->add_entryvar(smap.lookup(oev->origin()));
				auto nlv = ntheta->add_loopvar(ev->argument(1));
				r1map.insert(oev->argument(1), nlv->argument());
				nlvs[oev] = nlv;
			}
		}

		/* copy repetition region  */
		osubregion1->copy(ntheta->subregion(), r1map, false, false);

		/* adjust values in substitution map for condition node copying */
		for (const auto & olv : *otheta) {
			auto output = to_structural_output(olv->result()->origin());
			auto substitute = r1map.lookup(osubregion1->result(output->index())->origin());
			r1map.insert(olv->argument(), substitute);
		}

		/* copy condition nodes */
		copy_condition_nodes(ntheta->subregion(), r1map, cnodes);
		auto predicate = r1map.lookup(ogamma->predicate()->origin());

		/* redirect results of loop variables and adjust substitution map for exit region copying */
		for (const auto & olv : *otheta) {
			auto output = to_structural_output(olv->result()->origin());
			auto substitute = r1map.lookup(osubregion1->result(output->index())->origin());
			nlvs[olv->input()]->result()->divert_to(substitute);
			r1map.insert(olv->result()->origin(), nlvs[olv->input()]);
		}
		for (size_t n = 1; n < ogamma->ninputs(); n++) {
			auto oev = static_cast<jive::gamma_input*>(ogamma->input(n));
			if (auto argument = to_argument(oev->origin())) {
				r1map.insert(oev->argument(0), nlvs[argument->input()]);
			} else {
				auto substitute = r1map.lookup(oev->origin());
				nlvs[oev]->result()->divert_to(substitute);
				r1map.insert(oev->argument(0), nlvs[oev]);
			}
		}

		ntheta->set_predicate(predicate);

		/* copy exit region */
		osubregion0->copy(ngamma->subregion(1), r1map, false, false);

		/* adjust values in substitution map for exit variable creation */
		for (const auto & olv : *otheta) {
			auto output = to_structural_output(olv->result()->origin());
			auto substitute = r1map.lookup(osubregion0->result(output->index())->origin());
			r1map.insert(olv->result()->origin(), substitute);
		}

	}

	/* add exit variables to new gamma */
	for (const auto & olv : *otheta) {
		auto o0 = r0map.lookup(olv->result()->origin());
		auto o1 = r1map.lookup(olv->result()->origin());
		auto ex = ngamma->add_exitvar({o0, o1});
		smap.insert(olv, ex);
	}

	/* replace outputs */
	for (const auto & olv : *otheta)
		olv->divert_users(smap.lookup(olv));
	remove(otheta);
}

static void
invert(jive::region * region)
{
	for (auto & node : jive::topdown_traverser(region)) {
		if (auto structnode = dynamic_cast<jive::structural_node*>(node)) {
			for (size_t r = 0; r < structnode->nsubregions(); r++)
				invert(structnode->subregion(r));

			if (auto theta = dynamic_cast<jive::theta_node*>(structnode))
				invert(theta);
		}
	}
}

static void
invert(
  RvsdgModule & rm,
  StatisticsCollector & statisticsCollector)
{
	auto statistics = ivtstat::Create();

	statistics->start(rm.Rvsdg());
	invert(rm.Rvsdg().root());
	statistics->end(rm.Rvsdg());

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

/* tginversion */

tginversion::~tginversion()
{}

void
tginversion::run(
  RvsdgModule & module,
  StatisticsCollector & statisticsCollector)
{
	invert(module, statisticsCollector);
}

}

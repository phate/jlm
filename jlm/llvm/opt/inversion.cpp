/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/inversion.hpp>
#include <jlm/llvm/opt/pull.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::llvm
{

class ivtstat final : public util::Statistics
{
public:
  ~ivtstat() override = default;

  explicit ivtstat(const util::filepath & sourceFile)
      : Statistics(Statistics::Id::ThetaGammaInversion, sourceFile)
  {}

  void
  start(const jlm::rvsdg::graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(graph.root()));
    AddMeasurement(Label::NumRvsdgInputsBefore, rvsdg::ninputs(graph.root()));
    AddTimer(Label::Timer).start();
  }

  void
  end(const jlm::rvsdg::graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesAfter, rvsdg::nnodes(graph.root()));
    AddMeasurement(Label::NumRvsdgInputsAfter, rvsdg::ninputs(graph.root()));
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<ivtstat>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<ivtstat>(sourceFile);
  }
};

static jlm::rvsdg::gamma_node *
is_applicable(const jlm::rvsdg::theta_node * theta)
{
  auto matchnode = jlm::rvsdg::node_output::node(theta->predicate()->origin());
  if (!jlm::rvsdg::is<jlm::rvsdg::match_op>(matchnode))
    return nullptr;

  if (matchnode->output(0)->nusers() != 2)
    return nullptr;

  jlm::rvsdg::gamma_node * gnode = nullptr;
  for (const auto & user : *matchnode->output(0))
  {
    if (user == theta->predicate())
      continue;

    if (!rvsdg::is<rvsdg::gamma_op>(rvsdg::input::GetNode(*user)))
      return nullptr;

    gnode = dynamic_cast<rvsdg::gamma_node *>(rvsdg::input::GetNode(*user));
  }

  return gnode;
}

static void
pullin(jlm::rvsdg::gamma_node * gamma, jlm::rvsdg::theta_node * theta)
{
  pullin_bottom(gamma);
  for (const auto & lv : *theta)
  {
    if (jlm::rvsdg::node_output::node(lv->result()->origin()) != gamma)
    {
      auto ev = gamma->add_entryvar(lv->result()->origin());
      JLM_ASSERT(ev->narguments() == 2);
      auto xv = gamma->add_exitvar({ ev->argument(0), ev->argument(1) });
      lv->result()->divert_to(xv);
    }
  }
  pullin_top(gamma);
}

static std::vector<std::vector<jlm::rvsdg::node *>>
collect_condition_nodes(jlm::rvsdg::structural_node * tnode, jlm::rvsdg::structural_node * gnode)
{
  JLM_ASSERT(jlm::rvsdg::is<jlm::rvsdg::theta_op>(tnode));
  JLM_ASSERT(jlm::rvsdg::is<jlm::rvsdg::gamma_op>(gnode));
  JLM_ASSERT(gnode->region()->node() == tnode);

  std::vector<std::vector<jlm::rvsdg::node *>> nodes;
  for (auto & node : tnode->subregion(0)->nodes)
  {
    if (&node == gnode)
      continue;

    if (node.depth() >= nodes.size())
      nodes.resize(node.depth() + 1);
    nodes[node.depth()].push_back(&node);
  }

  return nodes;
}

static void
copy_condition_nodes(
    jlm::rvsdg::region * target,
    jlm::rvsdg::substitution_map & smap,
    const std::vector<std::vector<jlm::rvsdg::node *>> & nodes)
{
  for (size_t n = 0; n < nodes.size(); n++)
  {
    for (const auto & node : nodes[n])
      node->copy(target, smap);
  }
}

static jlm::rvsdg::argument *
to_argument(jlm::rvsdg::output * output)
{
  return dynamic_cast<jlm::rvsdg::argument *>(output);
}

static jlm::rvsdg::structural_output *
to_structural_output(jlm::rvsdg::output * output)
{
  return dynamic_cast<jlm::rvsdg::structural_output *>(output);
}

static void
invert(jlm::rvsdg::theta_node * otheta)
{
  auto ogamma = is_applicable(otheta);
  if (!ogamma)
    return;

  pullin(ogamma, otheta);

  /* copy condition nodes for new gamma node */
  jlm::rvsdg::substitution_map smap;
  auto cnodes = collect_condition_nodes(otheta, ogamma);
  for (const auto & olv : *otheta)
    smap.insert(olv->argument(), olv->input()->origin());
  copy_condition_nodes(otheta->region(), smap, cnodes);

  auto ngamma = jlm::rvsdg::gamma_node::create(
      smap.lookup(ogamma->predicate()->origin()),
      ogamma->nsubregions());

  /* handle subregion 0 */
  jlm::rvsdg::substitution_map r0map;
  {
    /* setup substitution map for exit region copying */
    auto osubregion0 = ogamma->subregion(0);
    for (auto oev = ogamma->begin_entryvar(); oev != ogamma->end_entryvar(); oev++)
    {
      if (auto argument = to_argument(oev->origin()))
      {
        auto nev = ngamma->add_entryvar(argument->input()->origin());
        r0map.insert(oev->argument(0), nev->argument(0));
      }
      else
      {
        auto substitute = smap.lookup(oev->origin());
        auto nev = ngamma->add_entryvar(substitute);
        r0map.insert(oev->argument(0), nev->argument(0));
      }
    }

    /* copy exit region */
    osubregion0->copy(ngamma->subregion(0), r0map, false, false);

    /* update substitution map for insertion of exit variables */
    for (const auto & olv : *otheta)
    {
      auto output = to_structural_output(olv->result()->origin());
      auto substitute = r0map.lookup(osubregion0->result(output->index())->origin());
      r0map.insert(olv->result()->origin(), substitute);
    }
  }

  /* handle subregion 1 */
  jlm::rvsdg::substitution_map r1map;
  {
    auto ntheta = jlm::rvsdg::theta_node::create(ngamma->subregion(1));

    /* add loop variables to new theta node and setup substitution map */
    auto osubregion0 = ogamma->subregion(0);
    auto osubregion1 = ogamma->subregion(1);
    std::unordered_map<jlm::rvsdg::input *, jlm::rvsdg::theta_output *> nlvs;
    for (const auto & olv : *otheta)
    {
      auto ev = ngamma->add_entryvar(olv->input()->origin());
      auto nlv = ntheta->add_loopvar(ev->argument(1));
      r1map.insert(olv->argument(), nlv->argument());
      nlvs[olv->input()] = nlv;
    }
    for (size_t n = 1; n < ogamma->ninputs(); n++)
    {
      auto oev = static_cast<jlm::rvsdg::gamma_input *>(ogamma->input(n));
      if (auto argument = to_argument(oev->origin()))
      {
        r1map.insert(oev->argument(1), nlvs[argument->input()]->argument());
      }
      else
      {
        auto ev = ngamma->add_entryvar(smap.lookup(oev->origin()));
        auto nlv = ntheta->add_loopvar(ev->argument(1));
        r1map.insert(oev->argument(1), nlv->argument());
        nlvs[oev] = nlv;
      }
    }

    /* copy repetition region  */
    osubregion1->copy(ntheta->subregion(), r1map, false, false);

    /* adjust values in substitution map for condition node copying */
    for (const auto & olv : *otheta)
    {
      auto output = to_structural_output(olv->result()->origin());
      auto substitute = r1map.lookup(osubregion1->result(output->index())->origin());
      r1map.insert(olv->argument(), substitute);
    }

    /* copy condition nodes */
    copy_condition_nodes(ntheta->subregion(), r1map, cnodes);
    auto predicate = r1map.lookup(ogamma->predicate()->origin());

    /* redirect results of loop variables and adjust substitution map for exit region copying */
    for (const auto & olv : *otheta)
    {
      auto output = to_structural_output(olv->result()->origin());
      auto substitute = r1map.lookup(osubregion1->result(output->index())->origin());
      nlvs[olv->input()]->result()->divert_to(substitute);
      r1map.insert(olv->result()->origin(), nlvs[olv->input()]);
    }
    for (size_t n = 1; n < ogamma->ninputs(); n++)
    {
      auto oev = static_cast<jlm::rvsdg::gamma_input *>(ogamma->input(n));
      if (auto argument = to_argument(oev->origin()))
      {
        r1map.insert(oev->argument(0), nlvs[argument->input()]);
      }
      else
      {
        auto substitute = r1map.lookup(oev->origin());
        nlvs[oev]->result()->divert_to(substitute);
        r1map.insert(oev->argument(0), nlvs[oev]);
      }
    }

    ntheta->set_predicate(predicate);

    /* copy exit region */
    osubregion0->copy(ngamma->subregion(1), r1map, false, false);

    /* adjust values in substitution map for exit variable creation */
    for (const auto & olv : *otheta)
    {
      auto output = to_structural_output(olv->result()->origin());
      auto substitute = r1map.lookup(osubregion0->result(output->index())->origin());
      r1map.insert(olv->result()->origin(), substitute);
    }
  }

  /* add exit variables to new gamma */
  for (const auto & olv : *otheta)
  {
    auto o0 = r0map.lookup(olv->result()->origin());
    auto o1 = r1map.lookup(olv->result()->origin());
    auto ex = ngamma->add_exitvar({ o0, o1 });
    smap.insert(olv, ex);
  }

  /* replace outputs */
  for (const auto & olv : *otheta)
    olv->divert_users(smap.lookup(olv));
  remove(otheta);
}

static void
invert(jlm::rvsdg::region * region)
{
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto structnode = dynamic_cast<jlm::rvsdg::structural_node *>(node))
    {
      for (size_t r = 0; r < structnode->nsubregions(); r++)
        invert(structnode->subregion(r));

      if (auto theta = dynamic_cast<jlm::rvsdg::theta_node *>(structnode))
        invert(theta);
    }
  }
}

static void
invert(RvsdgModule & rm, util::StatisticsCollector & statisticsCollector)
{
  auto statistics = ivtstat::Create(rm.SourceFileName());

  statistics->start(rm.Rvsdg());
  invert(rm.Rvsdg().root());
  statistics->end(rm.Rvsdg());

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

/* tginversion */

tginversion::~tginversion()
{}

void
tginversion::run(RvsdgModule & module, jlm::util::StatisticsCollector & statisticsCollector)
{
  invert(module, statisticsCollector);
}

}

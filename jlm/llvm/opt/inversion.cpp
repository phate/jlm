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

static rvsdg::GammaNode *
is_applicable(const rvsdg::ThetaNode * theta)
{
  auto matchnode = jlm::rvsdg::output::GetNode(*theta->predicate()->origin());
  if (!jlm::rvsdg::is<jlm::rvsdg::match_op>(matchnode))
    return nullptr;

  if (matchnode->output(0)->nusers() != 2)
    return nullptr;

  rvsdg::GammaNode * gnode = nullptr;
  for (const auto & user : *matchnode->output(0))
  {
    if (user == theta->predicate())
      continue;

    if (!rvsdg::is<rvsdg::GammaOperation>(rvsdg::input::GetNode(*user)))
      return nullptr;

    gnode = dynamic_cast<rvsdg::GammaNode *>(rvsdg::input::GetNode(*user));
  }

  return gnode;
}

static void
pullin(rvsdg::GammaNode * gamma, rvsdg::ThetaNode * theta)
{
  pullin_bottom(gamma);
  for (const auto & lv : *theta)
  {
    if (jlm::rvsdg::output::GetNode(*lv->result()->origin()) != gamma)
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
collect_condition_nodes(rvsdg::StructuralNode * tnode, jlm::rvsdg::StructuralNode * gnode)
{
  JLM_ASSERT(is<rvsdg::ThetaOperation>(tnode));
  JLM_ASSERT(rvsdg::is<rvsdg::GammaOperation>(gnode));
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
    rvsdg::Region * target,
    rvsdg::SubstitutionMap & smap,
    const std::vector<std::vector<jlm::rvsdg::node *>> & nodes)
{
  for (size_t n = 0; n < nodes.size(); n++)
  {
    for (const auto & node : nodes[n])
      node->copy(target, smap);
  }
}

static rvsdg::RegionArgument *
to_argument(jlm::rvsdg::output * output)
{
  return dynamic_cast<rvsdg::RegionArgument *>(output);
}

static jlm::rvsdg::structural_output *
to_structural_output(jlm::rvsdg::output * output)
{
  return dynamic_cast<jlm::rvsdg::structural_output *>(output);
}

static void
invert(rvsdg::ThetaNode * otheta)
{
  auto ogamma = is_applicable(otheta);
  if (!ogamma)
    return;

  pullin(ogamma, otheta);

  /* copy condition nodes for new gamma node */
  rvsdg::SubstitutionMap smap;
  auto cnodes = collect_condition_nodes(otheta, ogamma);
  for (const auto & olv : *otheta)
    smap.insert(olv->argument(), olv->input()->origin());
  copy_condition_nodes(otheta->region(), smap, cnodes);

  auto ngamma =
      rvsdg::GammaNode::create(smap.lookup(ogamma->predicate()->origin()), ogamma->nsubregions());

  /* handle subregion 0 */
  rvsdg::SubstitutionMap r0map;
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
  rvsdg::SubstitutionMap r1map;
  {
    auto ntheta = rvsdg::ThetaNode::create(ngamma->subregion(1));

    /* add loop variables to new theta node and setup substitution map */
    auto osubregion0 = ogamma->subregion(0);
    auto osubregion1 = ogamma->subregion(1);
    std::unordered_map<jlm::rvsdg::input *, rvsdg::ThetaOutput *> nlvs;
    for (const auto & olv : *otheta)
    {
      auto ev = ngamma->add_entryvar(olv->input()->origin());
      auto nlv = ntheta->add_loopvar(ev->argument(1));
      r1map.insert(olv->argument(), nlv->argument());
      nlvs[olv->input()] = nlv;
    }
    for (size_t n = 1; n < ogamma->ninputs(); n++)
    {
      auto oev = util::AssertedCast<rvsdg::GammaInput>(ogamma->input(n));
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
      auto oev = util::AssertedCast<rvsdg::GammaInput>(ogamma->input(n));
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
invert(rvsdg::Region * region)
{
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      for (size_t r = 0; r < structnode->nsubregions(); r++)
        invert(structnode->subregion(r));

      if (auto theta = dynamic_cast<rvsdg::ThetaNode *>(structnode))
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

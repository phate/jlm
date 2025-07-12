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

  explicit ivtstat(const util::FilePath & sourceFile)
      : Statistics(Statistics::Id::ThetaGammaInversion, sourceFile)
  {}

  void
  start(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(&graph.GetRootRegion()));
    AddMeasurement(Label::NumRvsdgInputsBefore, rvsdg::ninputs(&graph.GetRootRegion()));
    AddTimer(Label::Timer).start();
  }

  void
  end(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesAfter, rvsdg::nnodes(&graph.GetRootRegion()));
    AddMeasurement(Label::NumRvsdgInputsAfter, rvsdg::ninputs(&graph.GetRootRegion()));
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<ivtstat>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<ivtstat>(sourceFile);
  }
};

static rvsdg::GammaNode *
is_applicable(const rvsdg::ThetaNode * theta)
{
  auto matchNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*theta->predicate()->origin());
  if (!jlm::rvsdg::is<jlm::rvsdg::match_op>(matchNode))
    return nullptr;

  if (matchNode->output(0)->nusers() != 2)
    return nullptr;

  rvsdg::GammaNode * gnode = nullptr;
  for (const auto & user : *matchNode->output(0))
  {
    if (user == theta->predicate())
      continue;

    gnode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(*user);
    if (!gnode)
      return nullptr;
  }
  // only apply tgi if theta is a converted for loop - i.e. everything but the predicate is
  // contained in the gamma
  for (auto & lv : theta->GetLoopVars())
  {
    auto origin = lv.post->origin();
    if (dynamic_cast<rvsdg::RegionArgument *>(origin))
    {
      // origin is a theta argument
    }
    else if (rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(*origin) == gnode)
    {
      // origin is gnode
    }
    else
    {
      // we don't want to invert this
      return nullptr;
    }
  }

  return gnode;
}

static void
pullin(rvsdg::GammaNode * gamma, rvsdg::ThetaNode * theta)
{
  pullin_bottom(gamma);
  for (const auto & lv : theta->GetLoopVars())
  {
    if (rvsdg::TryGetOwnerNode<rvsdg::Node>(*lv.post->origin()) != gamma)
    {
      auto ev = gamma->AddEntryVar(lv.post->origin());
      JLM_ASSERT(ev.branchArgument.size() == 2);
      auto xv = gamma->AddExitVar({ ev.branchArgument[0], ev.branchArgument[1] }).output;
      lv.post->divert_to(xv);
    }
  }
  pullin_top(gamma);
}

static std::vector<std::vector<rvsdg::Node *>>
collect_condition_nodes(rvsdg::StructuralNode * tnode, jlm::rvsdg::StructuralNode * gnode)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::ThetaNode *>(tnode));
  JLM_ASSERT(dynamic_cast<const rvsdg::GammaNode *>(gnode));
  JLM_ASSERT(gnode->region()->node() == tnode);

  std::vector<std::vector<rvsdg::Node *>> nodes;
  for (auto & node : tnode->subregion(0)->Nodes())
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
    const std::vector<std::vector<rvsdg::Node *>> & nodes)
{
  for (size_t n = 0; n < nodes.size(); n++)
  {
    for (const auto & node : nodes[n])
      node->copy(target, smap);
  }
}

static jlm::rvsdg::StructuralOutput *
to_structural_output(jlm::rvsdg::Output * output)
{
  return dynamic_cast<rvsdg::StructuralOutput *>(output);
}

static rvsdg::RegionArgument *
to_argument(jlm::rvsdg::Output * output)
{
  return dynamic_cast<rvsdg::RegionArgument *>(output);
}

static void
invert(rvsdg::ThetaNode * otheta)
{
  auto ogamma = is_applicable(otheta);
  if (!ogamma)
    return;

  NodeSinking(ogamma, otheta);

  /* copy condition nodes for new gamma node */
  rvsdg::SubstitutionMap smap;
  auto cnodes = collect_condition_nodes(otheta, ogamma);
  for (const auto & olv : otheta->GetLoopVars())
    smap.insert(olv.pre, olv.input->origin());
  copy_condition_nodes(otheta->region(), smap, cnodes);

  auto ngamma =
      rvsdg::GammaNode::create(smap.lookup(ogamma->predicate()->origin()), ogamma->nsubregions());

  /* handle subregion 0 */
  rvsdg::SubstitutionMap r0map;
  {
    /* setup substitution map for exit region copying */
    auto osubregion0 = ogamma->subregion(0);
    for (const auto & oev : ogamma->GetEntryVars())
    {
      if (auto argument = to_argument(oev.input->origin()))
      {
        auto nev = ngamma->AddEntryVar(argument->input()->origin());
        r0map.insert(oev.branchArgument[0], nev.branchArgument[0]);
      }
      else
      {
        auto substitute = smap.lookup(oev.input->origin());
        auto nev = ngamma->AddEntryVar(substitute);
        r0map.insert(oev.branchArgument[0], nev.branchArgument[0]);
      }
    }

    /* copy exit region */
    osubregion0->copy(ngamma->subregion(0), r0map, false, false);

    /* update substitution map for insertion of exit variables */
    for (const auto & olv : otheta->GetLoopVars())
    {
      auto output = to_structural_output(olv.post->origin());
      auto substitute = r0map.lookup(osubregion0->result(output->index())->origin());
      r0map.insert(olv.post->origin(), substitute);
    }
  }

  /* handle subregion 1 */
  rvsdg::SubstitutionMap r1map;
  {
    auto ntheta = rvsdg::ThetaNode::create(ngamma->subregion(1));

    /* add loop variables to new theta node and setup substitution map */
    auto osubregion0 = ogamma->subregion(0);
    auto osubregion1 = ogamma->subregion(1);
    std::unordered_map<jlm::rvsdg::Input *, rvsdg::ThetaNode::LoopVar> nlvs;
    for (const auto & olv : otheta->GetLoopVars())
    {
      auto ev = ngamma->AddEntryVar(olv.input->origin());
      auto nlv = ntheta->AddLoopVar(ev.branchArgument[1]);
      r1map.insert(olv.pre, nlv.pre);
      nlvs[olv.input] = nlv;
    }
    for (const auto & oev : ogamma->GetEntryVars())
    {
      if (auto argument = to_argument(oev.input->origin()))
      {
        r1map.insert(oev.branchArgument[1], nlvs[argument->input()].pre);
      }
      else
      {
        auto ev = ngamma->AddEntryVar(smap.lookup(oev.input->origin()));
        auto nlv = ntheta->AddLoopVar(ev.branchArgument[1]);
        r1map.insert(oev.branchArgument[1], nlv.pre);
        nlvs[oev.input] = nlv;
      }
    }

    /* copy repetition region  */
    osubregion1->copy(ntheta->subregion(), r1map, false, false);

    /* adjust values in substitution map for condition node copying */
    for (const auto & olv : otheta->GetLoopVars())
    {
      auto output = to_structural_output(olv.post->origin());
      auto substitute = r1map.lookup(osubregion1->result(output->index())->origin());
      r1map.insert(olv.pre, substitute);
    }

    /* copy condition nodes */
    copy_condition_nodes(ntheta->subregion(), r1map, cnodes);
    auto predicate = r1map.lookup(ogamma->predicate()->origin());

    /* redirect results of loop variables and adjust substitution map for exit region copying */
    for (const auto & olv : otheta->GetLoopVars())
    {
      auto output = to_structural_output(olv.post->origin());
      auto substitute = r1map.lookup(osubregion1->result(output->index())->origin());
      nlvs[olv.input].post->divert_to(substitute);
      r1map.insert(olv.post->origin(), nlvs[olv.input].output);
    }
    for (const auto & oev : ogamma->GetEntryVars())
    {
      if (auto argument = to_argument(oev.input->origin()))
      {
        r1map.insert(oev.branchArgument[0], nlvs[argument->input()].output);
      }
      else
      {
        auto substitute = r1map.lookup(oev.input->origin());
        nlvs[oev.input].post->divert_to(substitute);
        r1map.insert(oev.branchArgument[0], nlvs[oev.input].output);
      }
    }

    ntheta->set_predicate(predicate);

    /* copy exit region */
    osubregion0->copy(ngamma->subregion(1), r1map, false, false);

    /* adjust values in substitution map for exit variable creation */
    for (const auto & olv : otheta->GetLoopVars())
    {
      auto output = to_structural_output(olv.post->origin());
      auto substitute = r1map.lookup(osubregion0->result(output->index())->origin());
      r1map.insert(olv.post->origin(), substitute);
    }
  }

  /* add exit variables to new gamma */
  for (const auto & olv : otheta->GetLoopVars())
  {
    auto o0 = r0map.lookup(olv.post->origin());
    auto o1 = r1map.lookup(olv.post->origin());
    auto ex = ngamma->AddExitVar({ o0, o1 });
    smap.insert(olv.output, ex.output);
  }

  /* replace outputs */
  for (const auto & olv : otheta->GetLoopVars())
    olv.output->divert_users(smap.lookup(olv.output));
  remove(otheta);
}

static void
invert(rvsdg::Region * region)
{
  for (auto & node : rvsdg::TopDownTraverser(region))
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
invert(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
{
  auto statistics = ivtstat::Create(rvsdgModule.SourceFilePath().value());

  statistics->start(rvsdgModule.Rvsdg());
  invert(&rvsdgModule.Rvsdg().GetRootRegion());
  statistics->end(rvsdgModule.Rvsdg());

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

/* tginversion */

tginversion::~tginversion()
{}

void
tginversion::Run(rvsdg::RvsdgModule & module, jlm::util::StatisticsCollector & statisticsCollector)
{
  invert(module, statisticsCollector);
}

}

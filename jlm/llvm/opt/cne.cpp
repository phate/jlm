/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/cne.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

#include <typeindex>

namespace jlm::llvm
{

class CommonNodeElimination::Statistics final : public util::Statistics
{
  const char * MarkTimerLabel_ = "MarkTime";
  const char * DivertTimerLabel_ = "DivertTime";

public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Statistics::Id::CommonNodeElimination, sourceFile)
  {}

  void
  start_mark_stat(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(&graph.GetRootRegion()));
    AddMeasurement(Label::NumRvsdgInputsBefore, rvsdg::ninputs(&graph.GetRootRegion()));
    AddTimer(MarkTimerLabel_).start();
  }

  void
  end_mark_stat() noexcept
  {
    GetTimer(MarkTimerLabel_).stop();
  }

  void
  start_divert_stat() noexcept
  {
    AddTimer(DivertTimerLabel_).start();
  }

  void
  end_divert_stat(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesAfter, rvsdg::nnodes(&graph.GetRootRegion()));
    AddMeasurement(Label::NumRvsdgInputsAfter, rvsdg::ninputs(&graph.GetRootRegion()));
    GetTimer(DivertTimerLabel_).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

typedef std::unordered_set<jlm::rvsdg::Output *> congruence_set;

class CommonNodeElimination::Context final
{
public:
  inline void
  mark(jlm::rvsdg::Output * o1, jlm::rvsdg::Output * o2)
  {
    auto s1 = set(o1);
    auto s2 = set(o2);

    if (s1 == s2)
      return;

    if (s2->size() < s1->size())
    {
      s1 = outputs_[o2];
      s2 = outputs_[o1];
    }

    for (auto & o : *s1)
    {
      s2->insert(o);
      outputs_[o] = s2;
    }
  }

  inline void
  mark(const rvsdg::Node * n1, const rvsdg::Node * n2)
  {
    JLM_ASSERT(n1->noutputs() == n2->noutputs());

    for (size_t n = 0; n < n1->noutputs(); n++)
      mark(n1->output(n), n2->output(n));
  }

  inline bool
  congruent(jlm::rvsdg::Output * o1, jlm::rvsdg::Output * o2) const noexcept
  {
    if (o1 == o2)
      return true;

    auto it = outputs_.find(o1);
    if (it == outputs_.end())
      return false;

    return it->second->find(o2) != it->second->end();
  }

  inline bool
  congruent(const jlm::rvsdg::Input * i1, const jlm::rvsdg::Input * i2) const noexcept
  {
    return congruent(i1->origin(), i2->origin());
  }

  congruence_set *
  set(jlm::rvsdg::Output * output) noexcept
  {
    if (outputs_.find(output) == outputs_.end())
    {
      std::unique_ptr<congruence_set> set(new congruence_set({ output }));
      outputs_[output] = set.get();
      sets_.insert(std::move(set));
    }

    return outputs_[output];
  }

private:
  std::unordered_set<std::unique_ptr<congruence_set>> sets_;
  std::unordered_map<const jlm::rvsdg::Output *, congruence_set *> outputs_;
};

class VisitorSet final
{
public:
  void
  insert(const jlm::rvsdg::Output * o1, const jlm::rvsdg::Output * o2)
  {
    auto it = sets_.find(o1);
    if (it != sets_.end())
      sets_[o1].insert(o2);
    else
      sets_[o1] = { o2 };

    it = sets_.find(o2);
    if (it != sets_.end())
      sets_[o2].insert(o1);
    else
      sets_[o2] = { o1 };
  }

  bool
  visited(const jlm::rvsdg::Output * o1, const jlm::rvsdg::Output * o2) const
  {
    auto it = sets_.find(o1);
    if (it == sets_.end())
      return false;

    return it->second.find(o2) != it->second.end();
  }

private:
  std::unordered_map<const jlm::rvsdg::Output *, std::unordered_set<const jlm::rvsdg::Output *>>
      sets_;
};

static bool
congruent(
    rvsdg::Output * o1,
    rvsdg::Output * o2,
    VisitorSet & vs,
    CommonNodeElimination::Context & context)
{
  if (context.congruent(o1, o2) || vs.visited(o1, o2))
    return true;

  if (*o1->Type() != *o2->Type())
    return false;

  // Handle theta entry
  {
    const auto thetaNode1 = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*o1);
    const auto thetaNode2 = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*o2);
    if (thetaNode1 && thetaNode2)
    {
      vs.insert(o1, o2);
      const auto loopVariable1 = thetaNode1->MapPreLoopVar(*o1);
      const auto loopVariable2 = thetaNode2->MapPreLoopVar(*o2);

      if (!congruent(loopVariable1.input->origin(), loopVariable2.input->origin(), vs, context))
        return false;

      return congruent(loopVariable1.output, loopVariable2.output, vs, context);
    }
  }

  // Handle theta exit
  {
    const auto thetaNode1 = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*o1);
    const auto thetaNode2 = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*o2);
    if (thetaNode1 && thetaNode2)
    {
      vs.insert(o1, o2);
      const auto loopVariable1 = thetaNode1->MapOutputLoopVar(*o1);
      const auto loopVariable2 = thetaNode2->MapOutputLoopVar(*o2);

      if (rvsdg::ThetaLoopVarIsInvariant(loopVariable1)
          && rvsdg::ThetaLoopVarIsInvariant(loopVariable2))
      {
        // Both loop variables are invariant. This means both are always congruent even if they
        // are from different theta nodes with different iteration counts. In other words, it is
        // just a value that is passed through both thetas. Let's see whether both values are
        // congruent before they enter the loops.
        return congruent(loopVariable1.input->origin(), loopVariable2.input->origin(), vs, context);
      }

      if (thetaNode1 != thetaNode2)
      {
        // The loop variables are from different theta nodes. They would only be congruent if we can
        // ensure that both theta nodes have the same iteration count, but we do not want to invest
        // into this. Let's just bail out.
        return false;
      }

      return congruent(loopVariable1.post->origin(), loopVariable2.post->origin(), vs, context);
    }
  }

  // Handle gamma entry
  {
    const auto gammaNode1 = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(*o1);
    const auto gammaNode2 = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(*o2);
    if (gammaNode1 && gammaNode2)
    {
      JLM_ASSERT(gammaNode1 == gammaNode2);
      const auto origin1 = std::visit(
          [](const auto & roleVariable) -> rvsdg::Output *
          {
            return roleVariable.input->origin();
          },
          gammaNode1->MapBranchArgument(*o1));
      const auto origin2 = std::visit(
          [](const auto & roleVariable) -> rvsdg::Output *
          {
            return roleVariable.input->origin();
          },
          gammaNode2->MapBranchArgument(*o2));
      return congruent(origin1, origin2, vs, context);
    }
  }

  // Handle gamma exit
  {
    auto gammaNode1 = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(*o1);
    auto gammaNode2 = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(*o2);
    if (gammaNode1 && gammaNode2 && gammaNode1 == gammaNode2)
    {
      const auto [branchResults1, output1] = gammaNode1->MapOutputExitVar(*o1);
      const auto [branchResults2, output2] = gammaNode2->MapOutputExitVar(*o2);

      JLM_ASSERT(branchResults1.size() == branchResults2.size());
      for (size_t n = 0; n < branchResults1.size(); ++n)
      {
        JLM_ASSERT(branchResults1[n]->region() == branchResults2[n]->region());
        if (!congruent(branchResults1[n]->origin(), branchResults2[n]->origin(), vs, context))
          return false;
      }
      return true;
    }
  }

  // Handle simple nodes
  {
    const auto simpleNode1 = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*o1);
    const auto simpleNode2 = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*o2);
    if (simpleNode1 && simpleNode2 && simpleNode1->GetOperation() == simpleNode2->GetOperation()
        && simpleNode1->ninputs() == simpleNode2->ninputs() && o1->index() == o2->index())
    {
      for (auto & input : simpleNode1->Inputs())
      {
        const auto origin1 = input.origin();
        const auto origin2 = simpleNode2->input(input.index())->origin();
        if (!congruent(origin1, origin2, vs, context))
          return false;
      }
      return true;
    }
  }

  return false;
}

static bool
congruent(jlm::rvsdg::Output * o1, jlm::rvsdg::Output * o2, CommonNodeElimination::Context & ctx)
{
  VisitorSet vs;
  return congruent(o1, o2, vs, ctx);
}

static void
mark_arguments(
    rvsdg::StructuralInput * i1,
    rvsdg::StructuralInput * i2,
    CommonNodeElimination::Context & ctx)
{
  JLM_ASSERT(i1->node() && i1->node() == i2->node());
  JLM_ASSERT(i1->arguments.size() == i2->arguments.size());

  auto a1 = i1->arguments.begin();
  auto a2 = i2->arguments.begin();
  for (; a1 != i1->arguments.end(); a1++, a2++)
  {
    JLM_ASSERT(a1->region() == a2->region());
    if (congruent(a1.ptr(), a2.ptr(), ctx))
      ctx.mark(a1.ptr(), a2.ptr());
  }
}

static void
mark(rvsdg::Region *, CommonNodeElimination::Context & ctx);

static void
mark_gamma(const rvsdg::StructuralNode * node, CommonNodeElimination::Context & ctx)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::GammaNode *>(node));

  /* mark entry variables */
  for (size_t i1 = 1; i1 < node->ninputs(); i1++)
  {
    for (size_t i2 = i1 + 1; i2 < node->ninputs(); i2++)
      mark_arguments(node->input(i1), node->input(i2), ctx);
  }

  for (size_t n = 0; n < node->nsubregions(); n++)
    mark(node->subregion(n), ctx);

  /* mark exit variables */
  for (size_t o1 = 0; o1 < node->noutputs(); o1++)
  {
    for (size_t o2 = o1 + 1; o2 < node->noutputs(); o2++)
    {
      if (congruent(node->output(o1), node->output(o2), ctx))
        ctx.mark(node->output(o1), node->output(o2));
    }
  }
}

static void
mark_theta(const rvsdg::StructuralNode * node, CommonNodeElimination::Context & ctx)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::ThetaNode *>(node));
  auto theta = static_cast<const rvsdg::ThetaNode *>(node);

  /* mark loop variables */
  for (size_t i1 = 0; i1 < theta->ninputs(); i1++)
  {
    for (size_t i2 = i1 + 1; i2 < theta->ninputs(); i2++)
    {
      auto input1 = theta->input(i1);
      auto input2 = theta->input(i2);
      auto loopvar1 = theta->MapInputLoopVar(*input1);
      auto loopvar2 = theta->MapInputLoopVar(*input2);
      if (congruent(loopvar1.pre, loopvar2.pre, ctx))
      {
        ctx.mark(loopvar1.pre, loopvar2.pre);
        ctx.mark(loopvar1.output, loopvar2.output);
      }
    }
  }

  mark(node->subregion(0), ctx);
}

static void
mark_lambda(const rvsdg::StructuralNode * node, CommonNodeElimination::Context & ctx)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::LambdaNode *>(node));

  /* mark dependencies */
  for (size_t i1 = 0; i1 < node->ninputs(); i1++)
  {
    for (size_t i2 = i1 + 1; i2 < node->ninputs(); i2++)
    {
      auto input1 = node->input(i1);
      auto input2 = node->input(i2);
      if (ctx.congruent(input1, input2))
        ctx.mark(input1->arguments.first(), input2->arguments.first());
    }
  }

  mark(node->subregion(0), ctx);
}

static void
mark_phi(const rvsdg::StructuralNode * node, CommonNodeElimination::Context & ctx)
{
  auto & phi = *util::AssertedCast<const rvsdg::PhiNode>(node);

  /* mark dependencies */
  for (size_t i1 = 0; i1 < phi.ninputs(); i1++)
  {
    for (size_t i2 = i1 + 1; i2 < phi.ninputs(); i2++)
    {
      auto input1 = phi.input(i1);
      auto input2 = phi.input(i2);
      if (ctx.congruent(input1, input2))
        ctx.mark(input1->arguments.first(), input2->arguments.first());
    }
  }

  mark(phi.subregion(), ctx);
}

static void
mark_delta(const rvsdg::StructuralNode * node, CommonNodeElimination::Context &)
{
  JLM_ASSERT(rvsdg::is<DeltaOperation>(node));
}

static void
mark(const rvsdg::StructuralNode * node, CommonNodeElimination::Context & ctx)
{
  static std::unordered_map<
      std::type_index,
      void (*)(const rvsdg::StructuralNode *, CommonNodeElimination::Context &)>
      map({ { std::type_index(typeid(rvsdg::GammaNode)), mark_gamma },
            { std::type_index(typeid(rvsdg::ThetaNode)), mark_theta },
            { typeid(rvsdg::LambdaNode), mark_lambda },
            { typeid(rvsdg::PhiNode), mark_phi },
            { typeid(DeltaNode), mark_delta } });

  JLM_ASSERT(map.find(typeid(*node)) != map.end());
  map[typeid(*node)](node, ctx);
}

static void
mark(const jlm::rvsdg::SimpleNode * node, CommonNodeElimination::Context & ctx)
{
  if (node->ninputs() == 0)
  {
    for (const auto & other : node->region()->TopNodes())
    {
      if (&other != node && node->GetOperation() == other.GetOperation())
      {
        ctx.mark(node, &other);
        break;
      }
    }
    return;
  }

  auto set = ctx.set(node->input(0)->origin());
  for (const auto & origin : *set)
  {
    for (const auto & user : origin->Users())
    {
      const auto other = rvsdg::TryGetOwnerNode<rvsdg::Node>(user);
      if (!other || other == node || other->GetOperation() != node->GetOperation()
          || other->ninputs() != node->ninputs())
        continue;

      size_t n = 0;
      for (n = 0; n < node->ninputs(); n++)
      {
        if (!ctx.congruent(node->input(n), other->input(n)))
          break;
      }
      if (n == node->ninputs())
        ctx.mark(node, other);
    }
  }
}

static void
mark(rvsdg::Region * region, CommonNodeElimination::Context & ctx)
{
  for (const auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto simple = dynamic_cast<const jlm::rvsdg::SimpleNode *>(node))
      mark(simple, ctx);
    else
      mark(static_cast<const rvsdg::StructuralNode *>(node), ctx);
  }
}

/* divert phase */

static void
divert_users(jlm::rvsdg::Output * output, CommonNodeElimination::Context & ctx)
{
  auto set = ctx.set(output);
  for (auto & other : *set)
    other->divert_users(output);
  set->clear();
}

static void
divert_outputs(rvsdg::Node * node, CommonNodeElimination::Context & ctx)
{
  for (size_t n = 0; n < node->noutputs(); n++)
    divert_users(node->output(n), ctx);
}

static void
divert_arguments(rvsdg::Region * region, CommonNodeElimination::Context & ctx)
{
  for (size_t n = 0; n < region->narguments(); n++)
    divert_users(region->argument(n), ctx);
}

static void
divert(rvsdg::Region *, CommonNodeElimination::Context &);

static void
divert_gamma(rvsdg::StructuralNode * node, CommonNodeElimination::Context & ctx)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::GammaNode *>(node));
  auto gamma = static_cast<rvsdg::GammaNode *>(node);

  for (const auto & ev : gamma->GetEntryVars())
  {
    for (auto input : ev.branchArgument)
      divert_users(input, ctx);
  }

  for (size_t r = 0; r < node->nsubregions(); r++)
    divert(node->subregion(r), ctx);

  divert_outputs(node, ctx);
}

static void
divert_theta(rvsdg::StructuralNode * node, CommonNodeElimination::Context & ctx)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::ThetaNode *>(node));
  auto theta = static_cast<rvsdg::ThetaNode *>(node);
  auto subregion = node->subregion(0);

  for (const auto & lv : theta->GetLoopVars())
  {
    JLM_ASSERT(ctx.set(lv.pre)->size() == ctx.set(lv.output)->size());
    divert_users(lv.pre, ctx);
    divert_users(lv.output, ctx);
  }

  divert(subregion, ctx);
}

static void
divert_lambda(rvsdg::StructuralNode * node, CommonNodeElimination::Context & ctx)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::LambdaNode *>(node));

  divert_arguments(node->subregion(0), ctx);
  divert(node->subregion(0), ctx);
}

static void
divert_phi(rvsdg::StructuralNode * node, CommonNodeElimination::Context & ctx)
{
  auto & phi = *util::AssertedCast<const rvsdg::PhiNode>(node);

  divert_arguments(phi.subregion(), ctx);
  divert(phi.subregion(), ctx);
}

static void
divert_delta(rvsdg::StructuralNode * node, CommonNodeElimination::Context &)
{
  JLM_ASSERT(is<DeltaOperation>(node));
}

static void
divert(rvsdg::StructuralNode * node, CommonNodeElimination::Context & ctx)
{
  static std::unordered_map<
      std::type_index,
      void (*)(rvsdg::StructuralNode *, CommonNodeElimination::Context &)>
      map({ { std::type_index(typeid(rvsdg::GammaNode)), divert_gamma },
            { std::type_index(typeid(rvsdg::ThetaNode)), divert_theta },
            { typeid(rvsdg::LambdaNode), divert_lambda },
            { typeid(rvsdg::PhiNode), divert_phi },
            { typeid(DeltaNode), divert_delta } });

  JLM_ASSERT(map.find(typeid(*node)) != map.end());
  map[typeid(*node)](node, ctx);
}

static void
divert(rvsdg::Region * region, CommonNodeElimination::Context & ctx)
{
  for (const auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto simple = dynamic_cast<jlm::rvsdg::SimpleNode *>(node))
      divert_outputs(simple, ctx);
    else
      divert(static_cast<rvsdg::StructuralNode *>(node), ctx);
  }
}

CommonNodeElimination::~CommonNodeElimination() noexcept = default;

void
CommonNodeElimination::Run(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  const auto & rvsdg = module.Rvsdg();

  Context context;
  auto statistics = Statistics::Create(module.SourceFilePath().value());

  statistics->start_mark_stat(rvsdg);
  mark(&rvsdg.GetRootRegion(), context);
  statistics->end_mark_stat();

  statistics->start_divert_stat();
  divert(&rvsdg.GetRootRegion(), context);
  statistics->end_divert_stat(rvsdg);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

}

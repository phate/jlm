/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/ir/hls.hpp>
#include <jlm/hls/opt/cne.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

#include <typeindex>

namespace jlm::hls
{

using namespace jlm::rvsdg;

class CommonNodeElimination::Statistics final : public util::Statistics
{
  const char * MarkTimerLabel_ = "MarkTime";
  const char * DivertTimerLabel_ = "DivertTime";

public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Id::CommonNodeElimination, sourceFile)
  {}

  void
  start_mark_stat(const Graph & graph) noexcept
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
  end_divert_stat(const Graph & graph) noexcept
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

class Context final
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
  mark(const Node * n1, const Node * n2)
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

class vset
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

/* mark phase */

static bool
congruent(jlm::rvsdg::Output * o1, jlm::rvsdg::Output * o2, vset & vs, Context & ctx)
{
  if (ctx.congruent(o1, o2) || vs.visited(o1, o2))
    return true;

  if (*o1->Type() != *o2->Type())
    return false;

  if (auto theta1 = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*o1))
  {
    if (auto theta2 = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*o2))
    {
      JLM_ASSERT(o1->region()->node() == o2->region()->node());
      auto loopvar1 = theta1->MapPreLoopVar(*o1);
      auto loopvar2 = theta2->MapPreLoopVar(*o2);
      vs.insert(o1, o2);
      auto i1 = loopvar1.input, i2 = loopvar2.input;
      if (!congruent(loopvar1.input->origin(), loopvar2.input->origin(), vs, ctx))
        return false;

      auto output1 = o1->region()->node()->output(i1->index());
      auto output2 = o2->region()->node()->output(i2->index());
      return congruent(output1, output2, vs, ctx);
    }
  }

  if (auto theta1 = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*o1))
  {
    if (auto theta2 = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*o2))
    {
      if (theta1 == theta2)
      {
        vs.insert(o1, o2);
        auto loopvar1 = theta1->MapOutputLoopVar(*o1);
        auto loopvar2 = theta2->MapOutputLoopVar(*o2);
        auto r1 = loopvar1.post;
        auto r2 = loopvar2.post;
        return congruent(r1->origin(), r2->origin(), vs, ctx);
      }
    }
  }

  auto n1 = TryGetOwnerNode<Node>(*o1);
  auto n2 = TryGetOwnerNode<Node>(*o2);

  auto a1 = dynamic_cast<rvsdg::RegionArgument *>(o1);
  auto a2 = dynamic_cast<rvsdg::RegionArgument *>(o2);
  if (a1 && dynamic_cast<const LoopNode *>(a1->region()->node()) && a2
      && dynamic_cast<const LoopNode *>(a2->region()->node()))
  {
    JLM_ASSERT(o1->region()->node() == o2->region()->node());
    if (a1->input() && a2->input())
    {
      // input arguments
      vs.insert(a1, a2);
      return congruent(a1->input()->origin(), a2->input()->origin(), vs, ctx);
    }
  }

  if (dynamic_cast<const rvsdg::GammaNode *>(n1) && n1 == n2)
  {
    auto so1 = static_cast<StructuralOutput *>(o1);
    auto so2 = static_cast<StructuralOutput *>(o2);
    auto r1 = so1->results.begin();
    auto r2 = so2->results.begin();
    for (; r1 != so1->results.end(); r1++, r2++)
    {
      JLM_ASSERT(r1->region() == r2->region());
      if (!congruent(r1->origin(), r2->origin(), vs, ctx))
        return false;
    }
    return true;
  }

  if (auto g1 = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(*o1))
  {
    if (auto g2 = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(*o2))
    {
      JLM_ASSERT(g1 == g2);
      auto origin1 = std::visit(
          [](const auto & rolevar) -> rvsdg::Output *
          {
            return rolevar.input->origin();
          },
          g1->MapBranchArgument(*o1));
      auto origin2 = std::visit(
          [](const auto & rolevar) -> rvsdg::Output *
          {
            return rolevar.input->origin();
          },
          g2->MapBranchArgument(*o2));
      return congruent(origin1, origin2, vs, ctx);
    }
  }

  if (jlm::rvsdg::is<SimpleOperation>(n1) && jlm::rvsdg::is<SimpleOperation>(n2)
      && n1->GetOperation() == n2->GetOperation() && n1->ninputs() == n2->ninputs()
      && o1->index() == o2->index())
  {
    for (size_t n = 0; n < n1->ninputs(); n++)
    {
      auto origin1 = n1->input(n)->origin();
      auto origin2 = n2->input(n)->origin();
      if (!congruent(origin1, origin2, vs, ctx))
        return false;
    }
    return true;
  }

  return false;
}

static bool
congruent(jlm::rvsdg::Output * o1, jlm::rvsdg::Output * o2, Context & ctx)
{
  vset vs;
  return congruent(o1, o2, vs, ctx);
}

static void
mark_arguments(StructuralInput * i1, StructuralInput * i2, Context & ctx)
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
mark(jlm::rvsdg::Region *, Context &);

static void
mark_gamma(const rvsdg::StructuralNode * node, Context & ctx)
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
mark_theta(const rvsdg::StructuralNode * node, Context & ctx)
{
  JLM_ASSERT(dynamic_cast<const jlm::rvsdg::ThetaNode *>(node));
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
mark_loop(const rvsdg::StructuralNode * node, Context & ctx)
{
  JLM_ASSERT(dynamic_cast<const LoopNode *>(node));
  auto loop = static_cast<const LoopNode *>(node);

  /* mark loop variables */
  for (size_t i1 = 0; i1 < loop->ninputs(); i1++)
  {
    for (size_t i2 = i1 + 1; i2 < loop->ninputs(); i2++)
    {
      auto input1 = loop->input(i1);
      auto input2 = loop->input(i2);
      if (congruent(input1->arguments.first(), input2->arguments.first(), ctx))
      {
        ctx.mark(input1->arguments.first(), input2->arguments.first());
      }
    }
  }
  mark(node->subregion(0), ctx);
}

static void
mark_lambda(const rvsdg::StructuralNode * node, Context & ctx)
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
mark_phi(const rvsdg::StructuralNode * node, Context & ctx)
{
  auto phi = util::AssertedCast<const rvsdg::PhiNode>(node);

  auto ctxvars = phi->GetContextVars();

  /* mark dependencies */
  for (size_t i1 = 0; i1 < ctxvars.size(); ++i1)
  {
    for (size_t i2 = i1 + 1; i2 < ctxvars.size(); ++i2)
    {
      if (ctx.congruent(ctxvars[i1].input, ctxvars[i2].input))
      {
        ctx.mark(ctxvars[i1].inner, ctxvars[i2].inner);
      }
    }
  }

  mark(node->subregion(0), ctx);
}

static void
mark_delta(const rvsdg::StructuralNode * node, Context &)
{
  JLM_ASSERT(jlm::rvsdg::is<llvm::DeltaOperation>(node));
}

static void
mark(const rvsdg::StructuralNode * node, Context & ctx)
{
  static std::unordered_map<std::type_index, void (*)(const rvsdg::StructuralNode *, Context &)>
      map({ { std::type_index(typeid(GammaNode)), mark_gamma },
            { std::type_index(typeid(ThetaNode)), mark_theta },
            { std::type_index(typeid(LoopNode)), mark_loop },
            { typeid(LambdaNode), mark_lambda },
            { typeid(PhiNode), mark_phi },
            { typeid(rvsdg::DeltaNode), mark_delta } });

  JLM_ASSERT(map.find(typeid(*node)) != map.end());
  map[typeid(*node)](node, ctx);
}

static void
mark(const jlm::rvsdg::SimpleNode * node, Context & ctx)
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
      const auto other = TryGetOwnerNode<rvsdg::Node>(user);
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
mark(rvsdg::Region * region, Context & ctx)
{
  for (const auto & node : TopDownTraverser(region))
  {
    if (auto simple = dynamic_cast<const jlm::rvsdg::SimpleNode *>(node))
      mark(simple, ctx);
    else
      mark(static_cast<const rvsdg::StructuralNode *>(node), ctx);
  }
}

/* divert phase */

static void
divert_users(jlm::rvsdg::Output * output, Context & ctx)
{
  auto set = ctx.set(output);
  for (auto & other : *set)
    other->divert_users(output);
  set->clear();
}

static void
divert_outputs(Node * node, Context & ctx)
{
  for (size_t n = 0; n < node->noutputs(); n++)
    divert_users(node->output(n), ctx);
}

static void
divert_arguments(rvsdg::Region * region, Context & ctx)
{
  for (size_t n = 0; n < region->narguments(); n++)
    divert_users(region->argument(n), ctx);
}

static void
divert(rvsdg::Region *, Context &);

static void
divert_gamma(rvsdg::StructuralNode * node, Context & ctx)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::GammaNode *>(node));
  auto gamma = static_cast<GammaNode *>(node);

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
divert_theta(rvsdg::StructuralNode * node, Context & ctx)
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
divert_loop(rvsdg::StructuralNode * node, Context & ctx)
{
  JLM_ASSERT(dynamic_cast<const LoopNode *>(node));
  auto subregion = node->subregion(0);
  divert(subregion, ctx);
}

static void
divert_lambda(rvsdg::StructuralNode * node, Context & ctx)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::LambdaNode *>(node));

  divert_arguments(node->subregion(0), ctx);
  divert(node->subregion(0), ctx);
}

static void
divert_phi(rvsdg::StructuralNode * node, Context & ctx)
{
  auto phi = util::AssertedCast<PhiNode>(node);

  divert_arguments(phi->subregion(), ctx);
  divert(phi->subregion(), ctx);
}

static void
divert_delta(rvsdg::StructuralNode * node, Context &)
{
  JLM_ASSERT(jlm::rvsdg::is<llvm::DeltaOperation>(node));
}

static void
divert(rvsdg::StructuralNode * node, Context & ctx)
{
  static std::unordered_map<std::type_index, void (*)(rvsdg::StructuralNode *, Context &)> map(
      { { std::type_index(typeid(rvsdg::GammaNode)), divert_gamma },
        { std::type_index(typeid(ThetaNode)), divert_theta },
        { std::type_index(typeid(LoopNode)), divert_loop },
        { typeid(rvsdg::LambdaNode), divert_lambda },
        { typeid(PhiNode), divert_phi },
        { typeid(rvsdg::DeltaNode), divert_delta } });

  JLM_ASSERT(map.find(typeid(*node)) != map.end());
  map[typeid(*node)](node, ctx);
}

static void
divert(rvsdg::Region * region, Context & ctx)
{
  for (const auto & node : TopDownTraverser(region))
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
  const auto & graph = module.Rvsdg();

  Context ctx;
  auto statistics = Statistics::Create(module.SourceFilePath().value());

  statistics->start_mark_stat(graph);
  mark(&graph.GetRootRegion(), ctx);
  statistics->end_mark_stat();

  statistics->start_divert_stat();
  divert(&graph.GetRootRegion(), ctx);
  statistics->end_divert_stat(graph);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

}

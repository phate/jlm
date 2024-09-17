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

namespace jlm::hls
{

using namespace jlm::rvsdg;

class cnestat final : public util::Statistics
{
  const char * MarkTimerLabel_ = "MarkTime";
  const char * DivertTimerLabel_ = "DivertTime";

public:
  ~cnestat() override = default;

  explicit cnestat(const util::filepath & sourceFile)
      : Statistics(Statistics::Id::CommonNodeElimination, sourceFile)
  {}

  void
  start_mark_stat(const jlm::rvsdg::graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(graph.root()));
    AddMeasurement(Label::NumRvsdgInputsBefore, rvsdg::ninputs(graph.root()));
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
  end_divert_stat(const jlm::rvsdg::graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesAfter, rvsdg::nnodes(graph.root()));
    AddMeasurement(Label::NumRvsdgInputsAfter, rvsdg::ninputs(graph.root()));
    GetTimer(DivertTimerLabel_).stop();
  }

  static std::unique_ptr<cnestat>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<cnestat>(sourceFile);
  }
};

typedef std::unordered_set<jlm::rvsdg::output *> congruence_set;

class cnectx
{
public:
  inline void
  mark(jlm::rvsdg::output * o1, jlm::rvsdg::output * o2)
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
  mark(const jlm::rvsdg::node * n1, const jlm::rvsdg::node * n2)
  {
    JLM_ASSERT(n1->noutputs() == n2->noutputs());

    for (size_t n = 0; n < n1->noutputs(); n++)
      mark(n1->output(n), n2->output(n));
  }

  inline bool
  congruent(jlm::rvsdg::output * o1, jlm::rvsdg::output * o2) const noexcept
  {
    if (o1 == o2)
      return true;

    auto it = outputs_.find(o1);
    if (it == outputs_.end())
      return false;

    return it->second->find(o2) != it->second->end();
  }

  inline bool
  congruent(const jlm::rvsdg::input * i1, const jlm::rvsdg::input * i2) const noexcept
  {
    return congruent(i1->origin(), i2->origin());
  }

  congruence_set *
  set(jlm::rvsdg::output * output) noexcept
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
  std::unordered_map<const jlm::rvsdg::output *, congruence_set *> outputs_;
};

class vset
{
public:
  void
  insert(const jlm::rvsdg::output * o1, const jlm::rvsdg::output * o2)
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
  visited(const jlm::rvsdg::output * o1, const jlm::rvsdg::output * o2) const
  {
    auto it = sets_.find(o1);
    if (it == sets_.end())
      return false;

    return it->second.find(o2) != it->second.end();
  }

private:
  std::unordered_map<const jlm::rvsdg::output *, std::unordered_set<const jlm::rvsdg::output *>>
      sets_;
};

/* mark phase */

static bool
congruent(jlm::rvsdg::output * o1, jlm::rvsdg::output * o2, vset & vs, cnectx & ctx)
{
  if (ctx.congruent(o1, o2) || vs.visited(o1, o2))
    return true;

  if (o1->type() != o2->type())
    return false;

  if (is<rvsdg::ThetaArgument>(o1) && is<rvsdg::ThetaArgument>(o2))
  {
    JLM_ASSERT(o1->region()->node() == o2->region()->node());
    auto a1 = static_cast<rvsdg::RegionArgument *>(o1);
    auto a2 = static_cast<rvsdg::RegionArgument *>(o2);
    vs.insert(a1, a2);
    auto i1 = a1->input(), i2 = a2->input();
    if (!congruent(a1->input()->origin(), a2->input()->origin(), vs, ctx))
      return false;

    auto output1 = o1->region()->node()->output(i1->index());
    auto output2 = o2->region()->node()->output(i2->index());
    return congruent(output1, output2, vs, ctx);
  }

  auto n1 = jlm::rvsdg::node_output::node(o1);
  auto n2 = jlm::rvsdg::node_output::node(o2);
  if (is<jlm::rvsdg::ThetaOperation>(n1) && is<jlm::rvsdg::ThetaOperation>(n2) && n1 == n2)
  {
    auto so1 = static_cast<jlm::rvsdg::structural_output *>(o1);
    auto so2 = static_cast<jlm::rvsdg::structural_output *>(o2);
    vs.insert(o1, o2);
    auto r1 = so1->results.first();
    auto r2 = so2->results.first();
    return congruent(r1->origin(), r2->origin(), vs, ctx);
  }

  auto a1 = dynamic_cast<rvsdg::RegionArgument *>(o1);
  auto a2 = dynamic_cast<rvsdg::RegionArgument *>(o2);
  if (a1 && is<hls::loop_op>(a1->region()->node()) && a2 && is<hls::loop_op>(a2->region()->node()))
  {
    JLM_ASSERT(o1->region()->node() == o2->region()->node());
    if (a1->input() && a2->input())
    {
      // input arguments
      vs.insert(a1, a2);
      return congruent(a1->input()->origin(), a2->input()->origin(), vs, ctx);
    }
  }

  if (rvsdg::is<rvsdg::GammaOperation>(n1) && n1 == n2)
  {
    auto so1 = static_cast<jlm::rvsdg::structural_output *>(o1);
    auto so2 = static_cast<jlm::rvsdg::structural_output *>(o2);
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

  if (is<rvsdg::GammaArgument>(o1) && is<rvsdg::GammaArgument>(o2))
  {
    JLM_ASSERT(o1->region()->node() == o2->region()->node());
    auto a1 = static_cast<rvsdg::RegionArgument *>(o1);
    auto a2 = static_cast<rvsdg::RegionArgument *>(o2);
    return congruent(a1->input()->origin(), a2->input()->origin(), vs, ctx);
  }

  if (jlm::rvsdg::is<jlm::rvsdg::simple_op>(n1) && jlm::rvsdg::is<jlm::rvsdg::simple_op>(n2)
      && n1->operation() == n2->operation() && n1->ninputs() == n2->ninputs()
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
congruent(jlm::rvsdg::output * o1, jlm::rvsdg::output * o2, cnectx & ctx)
{
  vset vs;
  return congruent(o1, o2, vs, ctx);
}

static void
mark_arguments(jlm::rvsdg::structural_input * i1, jlm::rvsdg::structural_input * i2, cnectx & ctx)
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
mark(jlm::rvsdg::Region *, cnectx &);

static void
mark_gamma(const jlm::rvsdg::structural_node * node, cnectx & ctx)
{
  JLM_ASSERT(rvsdg::is<rvsdg::GammaOperation>(node->operation()));

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
mark_theta(const jlm::rvsdg::structural_node * node, cnectx & ctx)
{
  JLM_ASSERT(is<jlm::rvsdg::ThetaOperation>(node));
  auto theta = static_cast<const rvsdg::ThetaNode *>(node);

  /* mark loop variables */
  for (size_t i1 = 0; i1 < theta->ninputs(); i1++)
  {
    for (size_t i2 = i1 + 1; i2 < theta->ninputs(); i2++)
    {
      auto input1 = theta->input(i1);
      auto input2 = theta->input(i2);
      if (congruent(input1->argument(), input2->argument(), ctx))
      {
        ctx.mark(input1->argument(), input2->argument());
        ctx.mark(input1->output(), input2->output());
      }
    }
  }

  mark(node->subregion(0), ctx);
}

static void
mark_loop(const rvsdg::structural_node * node, cnectx & ctx)
{
  JLM_ASSERT(rvsdg::is<hls::loop_op>(node));
  auto loop = static_cast<const jlm::hls::loop_node *>(node);

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
mark_lambda(const jlm::rvsdg::structural_node * node, cnectx & ctx)
{
  JLM_ASSERT(jlm::rvsdg::is<llvm::lambda::operation>(node));

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
mark_phi(const jlm::rvsdg::structural_node * node, cnectx & ctx)
{
  JLM_ASSERT(is<llvm::phi::operation>(node));

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
mark_delta(const jlm::rvsdg::structural_node * node, cnectx & ctx)
{
  JLM_ASSERT(jlm::rvsdg::is<llvm::delta::operation>(node));
}

static void
mark(const jlm::rvsdg::structural_node * node, cnectx & ctx)
{
  static std::
      unordered_map<std::type_index, void (*)(const jlm::rvsdg::structural_node *, cnectx &)>
          map({ { std::type_index(typeid(rvsdg::GammaOperation)), mark_gamma },
                { std::type_index(typeid(ThetaOperation)), mark_theta },
                { std::type_index(typeid(jlm::hls::loop_op)), mark_loop },
                { typeid(llvm::lambda::operation), mark_lambda },
                { typeid(llvm::phi::operation), mark_phi },
                { typeid(llvm::delta::operation), mark_delta } });

  auto & op = node->operation();
  JLM_ASSERT(map.find(typeid(op)) != map.end());
  map[typeid(op)](node, ctx);
}

static void
mark(const jlm::rvsdg::simple_node * node, cnectx & ctx)
{
  if (node->ninputs() == 0)
  {
    for (const auto & other : node->region()->top_nodes)
    {
      if (&other != node && node->operation() == other.operation())
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
    for (const auto & user : *origin)
    {
      auto ni = dynamic_cast<const jlm::rvsdg::node_input *>(user);
      auto other = ni ? ni->node() : nullptr;
      if (!other || other == node || other->operation() != node->operation()
          || other->ninputs() != node->ninputs())
        continue;

      size_t n;
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
mark(rvsdg::Region * region, cnectx & ctx)
{
  for (const auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto simple = dynamic_cast<const jlm::rvsdg::simple_node *>(node))
      mark(simple, ctx);
    else
      mark(static_cast<const jlm::rvsdg::structural_node *>(node), ctx);
  }
}

/* divert phase */

static void
divert_users(jlm::rvsdg::output * output, cnectx & ctx)
{
  auto set = ctx.set(output);
  for (auto & other : *set)
    other->divert_users(output);
  set->clear();
}

static void
divert_outputs(jlm::rvsdg::node * node, cnectx & ctx)
{
  for (size_t n = 0; n < node->noutputs(); n++)
    divert_users(node->output(n), ctx);
}

static void
divert_arguments(rvsdg::Region * region, cnectx & ctx)
{
  for (size_t n = 0; n < region->narguments(); n++)
    divert_users(region->argument(n), ctx);
}

static void
divert(rvsdg::Region *, cnectx &);

static void
divert_gamma(jlm::rvsdg::structural_node * node, cnectx & ctx)
{
  JLM_ASSERT(rvsdg::is<rvsdg::GammaOperation>(node));
  auto gamma = static_cast<GammaNode *>(node);

  for (auto ev = gamma->begin_entryvar(); ev != gamma->end_entryvar(); ev++)
  {
    for (size_t n = 0; n < ev->narguments(); n++)
      divert_users(ev->argument(n), ctx);
  }

  for (size_t r = 0; r < node->nsubregions(); r++)
    divert(node->subregion(r), ctx);

  divert_outputs(node, ctx);
}

static void
divert_theta(jlm::rvsdg::structural_node * node, cnectx & ctx)
{
  JLM_ASSERT(is<jlm::rvsdg::ThetaOperation>(node));
  auto theta = static_cast<rvsdg::ThetaNode *>(node);
  auto subregion = node->subregion(0);

  for (const auto & lv : *theta)
  {
    JLM_ASSERT(ctx.set(lv->argument())->size() == ctx.set(lv)->size());
    divert_users(lv->argument(), ctx);
    divert_users(lv, ctx);
  }

  divert(subregion, ctx);
}

static void
divert_loop(rvsdg::structural_node * node, cnectx & ctx)
{
  JLM_ASSERT(rvsdg::is<hls::loop_op>(node));
  auto subregion = node->subregion(0);
  divert(subregion, ctx);
}

static void
divert_lambda(jlm::rvsdg::structural_node * node, cnectx & ctx)
{
  JLM_ASSERT(jlm::rvsdg::is<llvm::lambda::operation>(node));

  divert_arguments(node->subregion(0), ctx);
  divert(node->subregion(0), ctx);
}

static void
divert_phi(jlm::rvsdg::structural_node * node, cnectx & ctx)
{
  JLM_ASSERT(is<llvm::phi::operation>(node));

  divert_arguments(node->subregion(0), ctx);
  divert(node->subregion(0), ctx);
}

static void
divert_delta(jlm::rvsdg::structural_node * node, cnectx & ctx)
{
  JLM_ASSERT(jlm::rvsdg::is<llvm::delta::operation>(node));
}

static void
divert(jlm::rvsdg::structural_node * node, cnectx & ctx)
{
  static std::unordered_map<std::type_index, void (*)(jlm::rvsdg::structural_node *, cnectx &)> map(
      { { std::type_index(typeid(rvsdg::GammaOperation)), divert_gamma },
        { std::type_index(typeid(ThetaOperation)), divert_theta },
        { std::type_index(typeid(jlm::hls::loop_op)), divert_loop },
        { typeid(llvm::lambda::operation), divert_lambda },
        { typeid(llvm::phi::operation), divert_phi },
        { typeid(llvm::delta::operation), divert_delta } });

  auto & op = node->operation();
  JLM_ASSERT(map.find(typeid(op)) != map.end());
  map[typeid(op)](node, ctx);
}

static void
divert(rvsdg::Region * region, cnectx & ctx)
{
  for (const auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto simple = dynamic_cast<jlm::rvsdg::simple_node *>(node))
      divert_outputs(simple, ctx);
    else
      divert(static_cast<jlm::rvsdg::structural_node *>(node), ctx);
  }
}

static void
cne(jlm::llvm::RvsdgModule & rm, util::StatisticsCollector & statisticsCollector)
{
  auto & graph = rm.Rvsdg();

  cnectx ctx;
  auto statistics = cnestat::Create(rm.SourceFileName());

  statistics->start_mark_stat(graph);
  mark(graph.root(), ctx);
  statistics->end_mark_stat();

  statistics->start_divert_stat();
  divert(graph.root(), ctx);
  statistics->end_divert_stat(graph);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

/* cne class */

cne::~cne()
{}

void
cne::run(llvm::RvsdgModule & module, util::StatisticsCollector & statisticsCollector)
{
  hls::cne(module, statisticsCollector);
}

}

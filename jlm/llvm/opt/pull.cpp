/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/pull.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::llvm
{

class pullstat final : public util::Statistics
{
public:
  ~pullstat() override = default;

  explicit pullstat(const util::filepath & sourceFile)
      : Statistics(Statistics::Id::PullNodes, sourceFile)
  {}

  void
  start(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgInputsBefore, rvsdg::ninputs(&graph.GetRootRegion()));
    AddTimer(Label::Timer).start();
  }

  void
  end(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgInputsAfter, rvsdg::ninputs(&graph.GetRootRegion()));
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<pullstat>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<pullstat>(sourceFile);
  }
};

static bool
empty(const rvsdg::GammaNode * gamma)
{
  for (size_t n = 0; n < gamma->nsubregions(); n++)
  {
    if (gamma->subregion(n)->nnodes() != 0)
      return false;
  }

  return true;
}

static bool
single_successor(const rvsdg::Node * node)
{
  std::unordered_set<rvsdg::Node *> successors;
  for (size_t n = 0; n < node->noutputs(); n++)
  {
    for (const auto & user : *node->output(n))
      successors.insert(rvsdg::TryGetOwnerNode<rvsdg::Node>(*user));
  }

  return successors.size() == 1;
}

static void
pullin_node(rvsdg::GammaNode * gamma, rvsdg::Node * node)
{
  /* collect operands */
  std::vector<std::vector<jlm::rvsdg::output *>> operands(gamma->nsubregions());
  for (size_t i = 0; i < node->ninputs(); i++)
  {
    auto ev = gamma->AddEntryVar(node->input(i)->origin());
    std::size_t index = 0;
    for (auto input : ev.branchArgument)
      operands[index++].push_back(input);
  }

  /* copy node into subregions */
  for (size_t r = 0; r < gamma->nsubregions(); r++)
  {
    auto copy = node->copy(gamma->subregion(r), operands[r]);

    /* redirect outputs */
    for (size_t o = 0; o < node->noutputs(); o++)
    {
      for (const auto & user : *node->output(o))
      {
        auto entryvar = std::get<rvsdg::GammaNode::EntryVar>(gamma->MapInput(*user));
        entryvar.branchArgument[r]->divert_users(copy->output(o));
      }
    }
  }
}

static void
cleanup(rvsdg::GammaNode * gamma, rvsdg::Node * node)
{
  JLM_ASSERT(single_successor(node));

  /* remove entry variables and node */
  std::vector<rvsdg::GammaNode::EntryVar> entryvars;
  for (size_t n = 0; n < node->noutputs(); n++)
  {
    for (auto user : *node->output(n))
    {
      entryvars.push_back(std::get<rvsdg::GammaNode::EntryVar>(gamma->MapInput(*user)));
    }
  }
  gamma->RemoveEntryVars(entryvars);
  remove(node);
}

void
pullin_top(rvsdg::GammaNode * gamma)
{
  /* FIXME: This is inefficient. We can do better. */
  auto evs = gamma->GetEntryVars();
  size_t index = 0;
  while (index < evs.size())
  {
    const auto & ev = evs[index];
    auto node = rvsdg::TryGetOwnerNode<rvsdg::Node>(*ev.input->origin());
    auto tmp = rvsdg::TryGetOwnerNode<rvsdg::Node>(*gamma->predicate()->origin());
    if (node && tmp != node && single_successor(node))
    {
      pullin_node(gamma, node);

      cleanup(gamma, node);

      evs = gamma->GetEntryVars();
      index = 0;
    }
    else
    {
      index++;
    }
  }
}

void
pullin_bottom(rvsdg::GammaNode * gamma)
{
  /* collect immediate successors of the gamma node */
  std::unordered_set<rvsdg::Node *> workset;
  for (size_t n = 0; n < gamma->noutputs(); n++)
  {
    auto output = gamma->output(n);
    for (const auto & user : *output)
    {
      auto node = rvsdg::TryGetOwnerNode<rvsdg::Node>(*user);
      if (node && node->depth() == gamma->depth() + 1)
        workset.insert(node);
    }
  }

  while (!workset.empty())
  {
    auto node = *workset.begin();
    workset.erase(node);

    /* copy node into subregions */
    std::vector<std::vector<jlm::rvsdg::output *>> outputs(node->noutputs());
    for (size_t r = 0; r < gamma->nsubregions(); r++)
    {
      /* collect operands */
      std::vector<jlm::rvsdg::output *> operands;
      for (size_t i = 0; i < node->ninputs(); i++)
      {
        auto input = node->input(i);
        if (rvsdg::TryGetOwnerNode<rvsdg::Node>(*input->origin()) == gamma)
        {
          auto output = static_cast<rvsdg::StructuralOutput *>(input->origin());
          operands.push_back(gamma->subregion(r)->result(output->index())->origin());
        }
        else
        {
          auto ev = gamma->AddEntryVar(input->origin());
          operands.push_back(ev.branchArgument[r]);
        }
      }

      auto copy = node->copy(gamma->subregion(r), operands);
      for (size_t o = 0; o < copy->noutputs(); o++)
        outputs[o].push_back(copy->output(o));
    }

    /* adjust outputs and update workset */
    for (size_t n = 0; n < node->noutputs(); n++)
    {
      auto output = node->output(n);
      for (const auto & user : *output)
      {
        auto tmp = rvsdg::TryGetOwnerNode<rvsdg::Node>(*user);
        if (tmp && tmp->depth() == node->depth() + 1)
          workset.insert(tmp);
      }

      auto xv = gamma->AddExitVar(outputs[n]).output;
      output->divert_users(xv);
    }
  }
}

static size_t
is_used_in_nsubregions(const rvsdg::GammaNode * gamma, const rvsdg::Node * node)
{
  JLM_ASSERT(single_successor(node));

  /* collect all gamma inputs */
  std::unordered_set<const rvsdg::input *> inputs;
  for (size_t n = 0; n < node->noutputs(); n++)
  {
    for (const auto & user : *(node->output(n)))
    {
      inputs.insert(user);
    }
  }

  /* collect subregions where node is used */
  std::unordered_set<rvsdg::Region *> subregions;
  for (const auto & input : inputs)
  {
    std::visit(
        [&subregions](const auto & rolevar)
        {
          if constexpr (std::is_same<std::decay_t<decltype(rolevar)>, rvsdg::GammaNode::EntryVar>())
          {
            for (const auto & argument : rolevar.branchArgument)
            {
              if (argument->nusers() != 0)
                subregions.insert(argument->region());
            }
          }
          else if constexpr (std::is_same<
                                 std::decay_t<decltype(rolevar)>,
                                 rvsdg::GammaNode::MatchVar>())
          {
            for (const auto & argument : rolevar.matchContent)
            {
              if (argument->nusers() != 0)
                subregions.insert(argument->region());
            }
          }
          else
          {
            JLM_UNREACHABLE("A gamma input must either be the match variable or an entry variable");
          }
        },
        gamma->MapInput(*input));
  }

  return subregions.size();
}

void
pull(rvsdg::GammaNode * gamma)
{
  /*
    We don't want to pull anything into empty gammas with two subregions,
    as they are translated to select instructions in the r2j phase.
  */
  if (gamma->nsubregions() == 2 && empty(gamma))
    return;

  auto prednode = rvsdg::TryGetOwnerNode<rvsdg::Node>(*gamma->predicate()->origin());

  /* FIXME: This is inefficient. We can do better. */
  auto evs = gamma->GetEntryVars();
  size_t index = 0;
  while (index < evs.size())
  {
    const auto & ev = evs[index];
    auto node = rvsdg::TryGetOwnerNode<rvsdg::Node>(*ev.input->origin());
    if (!node || prednode == node || !single_successor(node))
    {
      index++;
      continue;
    }

    if (is_used_in_nsubregions(gamma, node) == 1)
    {
      /*
        FIXME: This function pulls in the node to ALL subregions and
        not just the one we care about.
      */
      pullin_node(gamma, node);
      cleanup(gamma, node);
      evs = gamma->GetEntryVars();
      index = 0;
    }
    else
    {
      index++;
    }
  }
}

void
pull(rvsdg::Region * region)
{
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      if (auto gamma = dynamic_cast<rvsdg::GammaNode *>(node))
        pull(gamma);

      for (size_t n = 0; n < structnode->nsubregions(); n++)
        pull(structnode->subregion(n));
    }
  }
}

static void
pull(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector)
{
  auto statistics = pullstat::Create(module.SourceFilePath().value());

  statistics->start(module.Rvsdg());
  pull(&module.Rvsdg().GetRootRegion());
  statistics->end(module.Rvsdg());

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

/* pullin class */

pullin::~pullin()
{}

void
pullin::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector)
{
  pull(module, statisticsCollector);
}

}

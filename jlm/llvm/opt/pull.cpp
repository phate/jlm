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

class NodeSinking::Statistics final : public util::Statistics
{
public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Statistics::Id::PullNodes, sourceFile)
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

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

static bool
empty(const rvsdg::GammaNode * gamma)
{
  for (size_t n = 0; n < gamma->nsubregions(); n++)
  {
    if (gamma->subregion(n)->numNodes() != 0)
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
    for (const auto & user : node->output(n)->Users())
      successors.insert(rvsdg::TryGetOwnerNode<rvsdg::Node>(user));
  }

  return successors.size() == 1;
}

static void
pullin_node(rvsdg::GammaNode * gamma, rvsdg::Node * node)
{
  /* collect operands */
  std::vector<std::vector<jlm::rvsdg::Output *>> operands(gamma->nsubregions());
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
      for (const auto & user : node->output(o)->Users())
      {
        auto entryvar = std::get<rvsdg::GammaNode::EntryVar>(gamma->MapInput(user));
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
    for (auto & user : node->output(n)->Users())
    {
      entryvars.push_back(std::get<rvsdg::GammaNode::EntryVar>(gamma->MapInput(user)));
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

util::HashSet<rvsdg::Node *>
NodeSinking::collectDependentNodes(const rvsdg::Node & node)
{
  std::function<void(const rvsdg::Node &, util::HashSet<rvsdg::Node *> &)> collect =
      [&collect](const rvsdg::Node & node, util::HashSet<rvsdg::Node *> & dependentNodes)
  {
    for (auto & output : node.Outputs())
    {
      for (auto & user : output.Users())
      {
        if (const auto userNode = rvsdg::TryGetOwnerNode<rvsdg::Node>(user))
        {
          if (dependentNodes.insert(userNode))
          {
            collect(*userNode, dependentNodes);
          }
        }
      }
    }
  };

  util::HashSet<rvsdg::Node *> dependentNodes;
  collect(node, dependentNodes);
  return dependentNodes;
}

std::vector<rvsdg::Node *>
NodeSinking::sortByDepth(const util::HashSet<rvsdg::Node *> & nodes)
{
  if (nodes.IsEmpty())
  {
    return {};
  }

  std::vector<rvsdg::Node *> sortedNodes;
  for (auto & node : nodes.Items())
  {
    sortedNodes.push_back(node);
  }

  auto depthMap = rvsdg::Region::computeDepthMap(*sortedNodes[0]->region());
  std::sort(
      sortedNodes.begin(),
      sortedNodes.end(),
      [&depthMap](const auto * node1, const auto * node2)
      {
        JLM_ASSERT(depthMap.find(node1) != depthMap.end());
        JLM_ASSERT(depthMap.find(node2) != depthMap.end());
        return depthMap[node1] < depthMap[node2];
      });

  return sortedNodes;
}

size_t
NodeSinking::sinkDependentNodesIntoGamma(rvsdg::GammaNode & gammaNode)
{
  const auto dependentNodes = collectDependentNodes(gammaNode);
  const auto sortedDependentNodes = sortByDepth(dependentNodes);

  // FIXME: We create too many entry and exit variables for the gamma node as we copy each
  // node individually instead of the entire subgraph. Rather copy the entire subgraph at once and
  // create entry and exit variables according to the "outer edges" of this subgraph.
  for (auto & node : sortedDependentNodes)
  {
    // Collect operands for each subregion we copy the node into
    std::unordered_map<rvsdg::Region *, std::vector<rvsdg::Output *>> subregionOperands;
    for (auto & input : node->Inputs())
    {
      auto & oldOperand = *input.origin();
      if (rvsdg::TryGetOwnerNode<rvsdg::Node>(oldOperand) == &gammaNode)
      {
        auto [branchResults, _] = gammaNode.MapOutputExitVar(oldOperand);
        for (const auto branchResult : branchResults)
        {
          subregionOperands[branchResult->region()].push_back(branchResult->origin());
        }
      }
      else
      {
        auto [_, branchArguments] = gammaNode.AddEntryVar(&oldOperand);
        for (auto branchArgument : branchArguments)
        {
          subregionOperands[branchArgument->region()].push_back(branchArgument);
        }
      }
    }

    // Copy node into each subregion and collect outputs of each copy
    std::unordered_map<rvsdg::Region *, std::vector<rvsdg::Output *>> subregionOutputs;
    for (auto & subregion : gammaNode.Subregions())
    {
      const auto copiedNode = node->copy(&subregion, subregionOperands.at(&subregion));
      subregionOutputs[&subregion] = rvsdg::outputs(copiedNode);
    }

    // Adjust outputs of original node
    for (auto & output : node->Outputs())
    {
      std::vector<rvsdg::Output *> branchResultOperands;
      for (auto & subregion : gammaNode.Subregions())
      {
        branchResultOperands.push_back(subregionOutputs[&subregion][output.index()]);
      }

      auto [_, exitVarOutput] = gammaNode.AddExitVar(branchResultOperands);
      output.divert_users(exitVarOutput);
    }
  }

  return dependentNodes.Size();
}

static size_t
is_used_in_nsubregions(const rvsdg::GammaNode * gamma, const rvsdg::Node * node)
{
  JLM_ASSERT(single_successor(node));

  /* collect all gamma inputs */
  std::unordered_set<const rvsdg::Input *> inputs;
  for (size_t n = 0; n < node->noutputs(); n++)
  {
    for (const auto & user : node->output(n)->Users())
    {
      inputs.insert(&user);
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
  auto statistics = NodeSinking::Statistics::Create(module.SourceFilePath().value());

  statistics->start(module.Rvsdg());
  pull(&module.Rvsdg().GetRootRegion());
  statistics->end(module.Rvsdg());

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

NodeSinking::~NodeSinking() noexcept = default;

void
NodeSinking::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector)
{
  pull(module, statisticsCollector);
}

}

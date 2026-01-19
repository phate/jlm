/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/LoopUnswitching.hpp>
#include <jlm/llvm/opt/pull.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::llvm
{

class LoopUnswitching::Statistics final : public util::Statistics
{
public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Id::LoopUnswitching, sourceFile)
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

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

rvsdg::GammaNode *
LoopUnswitching::IsUnswitchable(const rvsdg::ThetaNode & theta)
{
  auto [matchNode, matchOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::MatchOperation>(*theta.predicate()->origin());
  if (!matchOperation)
    return nullptr;

  // The output of the match node should only be connected to the theta and gamma node
  if (matchNode->output(0)->nusers() != 2)
    return nullptr;

  rvsdg::GammaNode * gammaNode = nullptr;
  for (const auto & user : matchNode->output(0)->Users())
  {
    if (&user == theta.predicate())
      continue;

    gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(user);
    if (!gammaNode)
      return nullptr;
  }

  // Only apply loop unswitching if the theta node is a converted for loop, i.e., everything but the
  // predicate is contained in the gamma
  for (const auto & loopVar : theta.GetLoopVars())
  {
    const auto origin = loopVar.post->origin();
    if (rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*origin))
    {
      // origin is a theta subregion argument
    }
    else if (rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(*origin) == gammaNode)
    {
      // origin is an output of gamma node
    }
    else
    {
      // we don't want to invert this
      return nullptr;
    }
  }

  return gammaNode;
}

void
LoopUnswitching::SinkNodesIntoGamma(
    rvsdg::GammaNode & gammaNode,
    const rvsdg::ThetaNode & thetaNode)
{
  NodeSinking::sinkDependentNodesIntoGamma(gammaNode);

  // Ensure all loop variables are routed through the gamma node
  for (const auto & loopVar : thetaNode.GetLoopVars())
  {
    if (rvsdg::TryGetOwnerNode<rvsdg::Node>(*loopVar.post->origin()) != &gammaNode)
    {
      auto [input, branchArgument] = gammaNode.AddEntryVar(loopVar.post->origin());
      JLM_ASSERT(branchArgument.size() == 2);
      auto [_, output] = gammaNode.AddExitVar({ branchArgument[0], branchArgument[1] });
      loopVar.post->divert_to(output);
    }
  }

  pullin_top(&gammaNode);
}

std::vector<std::vector<rvsdg::Node *>>
LoopUnswitching::CollectPredicateNodes(
    const rvsdg::ThetaNode & thetaNode,
    const rvsdg::GammaNode & gammaNode)
{
  JLM_ASSERT(gammaNode.region()->node() == &thetaNode);

  auto depthMap = rvsdg::computeDepthMap(*thetaNode.subregion());

  std::vector<std::vector<rvsdg::Node *>> nodes;
  for (auto & node : thetaNode.subregion()->Nodes())
  {
    if (&node == &gammaNode)
      continue;

    const auto depth = depthMap[&node];
    if (depth >= nodes.size())
      nodes.resize(depth + 1);
    nodes[depth].push_back(&node);
  }

  return nodes;
}

void
LoopUnswitching::CopyPredicateNodes(
    rvsdg::Region & target,
    rvsdg::SubstitutionMap & substitutionMap,
    const std::vector<std::vector<rvsdg::Node *>> & nodes)
{
  for (auto & sameDepthNodes : nodes)
  {
    for (const auto & node : sameDepthNodes)
      node->copy(&target, substitutionMap);
  }
}

rvsdg::SubstitutionMap
LoopUnswitching::handleGammaRepetitionRegion(
    rvsdg::ThetaNode & oldThetaNode,
    rvsdg::GammaNode & oldGammaNode,
    rvsdg::GammaNode & newGammaNode,
    const std::vector<std::vector<rvsdg::Node *>> & predicateNodes,
    const rvsdg::SubstitutionMap & substitutionMap)
{
  rvsdg::SubstitutionMap repetitionSubregionMap;

  auto newThetaNode = rvsdg::ThetaNode::create(newGammaNode.subregion(1));

  // Add loop variables to new theta node and setup substitution map
  auto exitSubregion = oldGammaNode.subregion(0);
  auto repetitionSubregion = oldGammaNode.subregion(1);

  std::unordered_map<rvsdg::Input *, rvsdg::ThetaNode::LoopVar> newLoopVars;
  for (const auto & oldLoopVar : oldThetaNode.GetLoopVars())
  {
    auto [_, branchArgument] = newGammaNode.AddEntryVar(oldLoopVar.input->origin());
    auto newLoopVar = newThetaNode->AddLoopVar(branchArgument[1]);
    repetitionSubregionMap.insert(oldLoopVar.pre, newLoopVar.pre);
    newLoopVars[oldLoopVar.input] = newLoopVar;
  }
  for (const auto & [oldInput, oldBranchArgument] : oldGammaNode.GetEntryVars())
  {
    if (rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*oldInput->origin()))
    {
      auto oldLoopVar = oldThetaNode.MapPreLoopVar(*oldInput->origin());
      repetitionSubregionMap.insert(oldBranchArgument[1], newLoopVars[oldLoopVar.input].pre);
    }
    else
    {
      auto [_, newBranchArgument] =
          newGammaNode.AddEntryVar(&substitutionMap.lookup(*oldInput->origin()));
      auto newLoopVar = newThetaNode->AddLoopVar(newBranchArgument[1]);
      repetitionSubregionMap.insert(oldBranchArgument[1], newLoopVar.pre);
      newLoopVars[oldInput] = newLoopVar;
    }
  }

  // Copy repetition region
  repetitionSubregion->copy(newThetaNode->subregion(), repetitionSubregionMap);

  // Adjust values in substitution map for condition node copying
  for (const auto & oldLopVar : oldThetaNode.GetLoopVars())
  {
    auto output = oldLopVar.post->origin();
    auto substitute =
        &repetitionSubregionMap.lookup(*repetitionSubregion->result(output->index())->origin());
    repetitionSubregionMap.insert(oldLopVar.pre, substitute);
  }

  // Copy condition nodes
  CopyPredicateNodes(*newThetaNode->subregion(), repetitionSubregionMap, predicateNodes);
  auto predicate = &repetitionSubregionMap.lookup(*oldGammaNode.predicate()->origin());

  // Redirect results of loop variables and adjust substitution map for exit region copying
  for (const auto & oldLoopVar : oldThetaNode.GetLoopVars())
  {
    auto output = oldLoopVar.post->origin();
    auto substitute =
        &repetitionSubregionMap.lookup(*repetitionSubregion->result(output->index())->origin());
    newLoopVars[oldLoopVar.input].post->divert_to(substitute);
    repetitionSubregionMap.insert(oldLoopVar.post->origin(), newLoopVars[oldLoopVar.input].output);
  }
  for (const auto & [input, branchArgument] : oldGammaNode.GetEntryVars())
  {
    if (rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*input->origin()))
    {
      auto oldLoopVar = oldThetaNode.MapPreLoopVar(*input->origin());
      repetitionSubregionMap.insert(branchArgument[0], newLoopVars[oldLoopVar.input].output);
    }
    else
    {
      auto substitute = &repetitionSubregionMap.lookup(*input->origin());
      newLoopVars[input].post->divert_to(substitute);
      repetitionSubregionMap.insert(branchArgument[0], newLoopVars[input].output);
    }
  }

  newThetaNode->set_predicate(predicate);

  // Copy exit region
  exitSubregion->copy(newGammaNode.subregion(1), repetitionSubregionMap);

  // Adjust values in substitution map for exit variable creation
  for (const auto & oldLoopVar : oldThetaNode.GetLoopVars())
  {
    auto output = oldLoopVar.post->origin();
    auto substitute =
        &repetitionSubregionMap.lookup(*exitSubregion->result(output->index())->origin());
    repetitionSubregionMap.insert(oldLoopVar.post->origin(), substitute);
  }

  return repetitionSubregionMap;
}

bool
LoopUnswitching::UnswitchLoop(rvsdg::ThetaNode & oldThetaNode)
{
  auto oldGammaNode = IsUnswitchable(oldThetaNode);
  if (!oldGammaNode)
    return false;

  SinkNodesIntoGamma(*oldGammaNode, oldThetaNode);

  // Copy condition nodes for new gamma node
  rvsdg::SubstitutionMap substitutionMap;
  auto conditionNodes = CollectPredicateNodes(oldThetaNode, *oldGammaNode);
  for (const auto & oldLoopVar : oldThetaNode.GetLoopVars())
    substitutionMap.insert(oldLoopVar.pre, oldLoopVar.input->origin());
  CopyPredicateNodes(*oldThetaNode.region(), substitutionMap, conditionNodes);

  auto newGammaNode = rvsdg::GammaNode::create(
      &substitutionMap.lookup(*oldGammaNode->predicate()->origin()),
      oldGammaNode->nsubregions());

  // Handle subregion 0
  rvsdg::SubstitutionMap subregion0Map;
  {
    // Setup substitution map for exit region copying
    auto oldSubregion0 = oldGammaNode->subregion(0);
    for (const auto & [oldInput, oldBranchArgument] : oldGammaNode->GetEntryVars())
    {
      if (rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*oldInput->origin()))
      {
        auto oldLoopVar = oldThetaNode.MapPreLoopVar(*oldInput->origin());
        auto [_, branchArgument] = newGammaNode->AddEntryVar(oldLoopVar.input->origin());
        subregion0Map.insert(oldBranchArgument[0], branchArgument[0]);
      }
      else
      {
        auto substitute = &substitutionMap.lookup(*oldInput->origin());
        auto [_, branchArgument] = newGammaNode->AddEntryVar(substitute);
        subregion0Map.insert(oldBranchArgument[0], branchArgument[0]);
      }
    }

    // Copy exit region
    oldSubregion0->copy(newGammaNode->subregion(0), subregion0Map);

    // Update substitution map for insertion of exit variables
    for (const auto & oldLoopVar : oldThetaNode.GetLoopVars())
    {
      auto output = oldLoopVar.post->origin();
      auto substitute = &subregion0Map.lookup(*oldSubregion0->result(output->index())->origin());
      subregion0Map.insert(oldLoopVar.post->origin(), substitute);
    }
  }

  auto repetitionSubstitutionMap = handleGammaRepetitionRegion(
      oldThetaNode,
      *oldGammaNode,
      *newGammaNode,
      conditionNodes,
      substitutionMap);

  // Add exit variables to new gamma
  for (const auto & oldLoopVar : oldThetaNode.GetLoopVars())
  {
    auto o0 = &subregion0Map.lookup(*oldLoopVar.post->origin());
    auto o1 = &repetitionSubstitutionMap.lookup(*oldLoopVar.post->origin());
    auto [_, output] = newGammaNode->AddExitVar({ o0, o1 });
    substitutionMap.insert(oldLoopVar.output, output);
  }

  // Replace outputs
  for (const auto & oldLoopVar : oldThetaNode.GetLoopVars())
    oldLoopVar.output->divert_users(&substitutionMap.lookup(*oldLoopVar.output));

  return true;
}

void
LoopUnswitching::HandleRegion(rvsdg::Region & region)
{
  bool unswitchedLoop = false;
  for (auto & node : region.Nodes())
  {
    if (const auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(&node))
    {
      // Handle innermost theta nodes first
      for (auto & subregion : structuralNode->Subregions())
        HandleRegion(subregion);

      if (const auto thetaNode = dynamic_cast<rvsdg::ThetaNode *>(structuralNode))
      {
        unswitchedLoop |= UnswitchLoop(*thetaNode);
      }
    }
  }

  // If we successfully unswitched a loop, ensure the old nodes are pruned.
  if (unswitchedLoop)
  {
    region.prune(false);
  }
}

LoopUnswitching::~LoopUnswitching() noexcept = default;

void
LoopUnswitching::Run(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFilePath().value());

  statistics->start(rvsdgModule.Rvsdg());
  HandleRegion(rvsdgModule.Rvsdg().GetRootRegion());
  statistics->end(rvsdgModule.Rvsdg());

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

void
LoopUnswitching::CreateAndRun(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  LoopUnswitching loopUnswitching;
  loopUnswitching.Run(rvsdgModule, statisticsCollector);
}

}

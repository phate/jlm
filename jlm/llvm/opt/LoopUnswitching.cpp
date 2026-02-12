/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/LoopUnswitching.hpp>
#include <jlm/llvm/opt/PredicateCorrelation.hpp>
#include <jlm/llvm/opt/pull.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::llvm
{

LoopUnswitchingHeuristic::~LoopUnswitchingHeuristic() noexcept = default;

LoopUnswitchingDefaultHeuristic::~LoopUnswitchingDefaultHeuristic() noexcept = default;

bool
LoopUnswitchingDefaultHeuristic::shouldUnswitchLoop(
    ThetaGammaPredicateCorrelation & correlation) const noexcept
{
  const auto & thetaNode = correlation.thetaNode();
  const auto & gammaNode = correlation.gammaNode();

  for (const auto node : rvsdg::TopDownConstTraverser(thetaNode.subregion()))
  {
    if (node == &gammaNode)
      continue;

    if (is<rvsdg::StructuralOperation>(node))
      return false;
  }

  return true;
}

std::shared_ptr<const LoopUnswitchingDefaultHeuristic>
LoopUnswitchingDefaultHeuristic::create()
{
  static const LoopUnswitchingDefaultHeuristic instance;
  return std::shared_ptr<const LoopUnswitchingDefaultHeuristic>(std::shared_ptr<void>(), &instance);
}

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

bool
LoopUnswitching::allLoopVarsAreRoutedThroughGamma(
    const rvsdg::ThetaNode & thetaNode,
    const rvsdg::GammaNode & gammaNode)
{
  for (const auto & loopVar : thetaNode.GetLoopVars())
  {
    const auto node = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(*loopVar.post->origin());
    if (node != &gammaNode)
      return false;
  }

  return true;
}

bool
LoopUnswitching::UnswitchLoop(rvsdg::ThetaNode & oldThetaNode)
{
  auto oldGammaNode = IsUnswitchable(oldThetaNode);
  if (!oldGammaNode)
    return false;

  SinkNodesIntoGamma(*oldGammaNode, oldThetaNode);

  // At this point, we have established the following invariant:
  // All loop variables of the original theta node are routed through its contained gamma node,
  // i.e., the origin of a loop variables' post value must be the output of the gamma node. This
  // helps to simplify the transformation significantly.
  JLM_ASSERT(allLoopVarsAreRoutedThroughGamma(oldThetaNode, *oldGammaNode));

  // FIXME: We should get this correlation from the IsUnswitchable() method, if it is possible
  // to perform the transformation.
  const auto correlationOpt = computeThetaGammaPredicateCorrelation(oldThetaNode);
  JLM_ASSERT(correlationOpt.has_value());
  auto & correlation = correlationOpt.value();

  if (!heuristic_->shouldUnswitchLoop(*correlation))
    return false;

  // The rest of the transformation is now performed in several stages:
  // 1. Copy the predicate subgraph into the parent region of the old theta node.
  // 2. Create gamma node (using the copied predicate) and the new theta node in the new gamma
  // nodes' repetition subregion.
  // 3. Copy old repetition subregion into the new theta node.
  // 4. Copy predicate subgraph into new theta node.
  // 5. Adjust the loop variables of the new theta node, finalizing the new theta node.
  // 6. Add exit variables to the new gamma node, finalizing the new gamma node.
  // 7. Copy old exit subregion into the parent region of the old theta node.
  // 8. Divert the users of old theta nodes' loop variables, rendering the old theta
  // node dead.
  //
  // Along the way, we keep track of replaced variables with substitution maps for some of the
  // stages. These substitution maps are then utilized by succeeding stages to find the correct
  // replacements for old outputs.

  // Stage 1 - Copy predicate subgraph into the old theta nodes' parent region
  rvsdg::SubstitutionMap stage1SMap;
  {
    for (const auto & oldLoopVar : oldThetaNode.GetLoopVars())
      stage1SMap.insert(oldLoopVar.pre, oldLoopVar.input->origin());

    auto conditionNodes = CollectPredicateNodes(oldThetaNode, *oldGammaNode);
    CopyPredicateNodes(*oldThetaNode.region(), stage1SMap, conditionNodes);
  }

  // Stage 2 - Create new gamma and theta node
  auto newGammaNode = rvsdg::GammaNode::create(
      &stage1SMap.lookup(*oldGammaNode->predicate()->origin()),
      oldGammaNode->nsubregions());

  const auto [oldRepetitionSubregion, oldExitSubregion] =
      determineGammaSubregionRoles(*correlation).value();
  const auto & newRepetitionSubregion = newGammaNode->subregion(oldRepetitionSubregion->index());
  const auto repetitionSubregionIndex = oldRepetitionSubregion->index();
  const auto exitSubregionIndex = oldExitSubregion->index();

  auto newThetaNode = rvsdg::ThetaNode::create(newRepetitionSubregion);

  std::unordered_map<rvsdg::Input *, rvsdg::Input *> oldGammaNewGammaInputMap;
  std::unordered_map<rvsdg::Input *, rvsdg::Input *> oldGammaNewThetaInputMap;
  for (const auto & [oldInput, oldBranchArgument] : oldGammaNode->GetEntryVars())
  {
    auto & newOrigin = stage1SMap.lookup(*oldInput->origin());
    auto newEntryVar = newGammaNode->AddEntryVar(&newOrigin);
    auto newLoopVar =
        newThetaNode->AddLoopVar(newEntryVar.branchArgument[repetitionSubregionIndex]);
    oldGammaNewGammaInputMap[oldInput] = newEntryVar.input;
    oldGammaNewThetaInputMap[oldInput] = newLoopVar.input;
  }

  // Stage 3 - Copy repetition subregion into new theta node
  rvsdg::SubstitutionMap stage3SMap;
  {
    for (const auto & [oldInput, oldBranchArgument] : oldGammaNode->GetEntryVars())
    {
      auto newLoopInput = oldGammaNewThetaInputMap[oldInput];
      auto newLoopVar = newThetaNode->MapInputLoopVar(*newLoopInput);
      stage3SMap.insert(oldBranchArgument[repetitionSubregionIndex], newLoopVar.pre);
    }

    oldRepetitionSubregion->copy(newThetaNode->subregion(), stage3SMap);
  }

  // Stage 4 - Copy predicate subgraph into new theta node subregion
  rvsdg::SubstitutionMap stage4SMap;
  {
    for (auto oldLoopVar : oldThetaNode.GetLoopVars())
    {
      auto oldExitVar = oldGammaNode->MapOutputExitVar(*oldLoopVar.post->origin());
      auto oldOrigin = oldExitVar.branchResult[repetitionSubregionIndex]->origin();
      auto & newOrigin = stage3SMap.lookup(*oldOrigin);
      stage4SMap.insert(oldLoopVar.pre, &newOrigin);
    }

    auto conditionNodes = CollectPredicateNodes(oldThetaNode, *oldGammaNode);
    CopyPredicateNodes(*newThetaNode->subregion(), stage4SMap, conditionNodes);
  }

  // Stage 5 - Adjust loop variables
  newThetaNode->set_predicate(&stage4SMap.lookup(*oldThetaNode.predicate()->origin()));
  for (const auto & [oldInput, oldBranchArgument] : oldGammaNode->GetEntryVars())
  {
    auto newLoopVarInput = oldGammaNewThetaInputMap[oldInput];
    auto newLoopVar = newThetaNode->MapInputLoopVar(*newLoopVarInput);
    auto & newOrigin = stage4SMap.lookup(*oldInput->origin());
    newLoopVar.post->divert_to(&newOrigin);
  }

  // Stage 6 - Add new gamma exit variables
  std::unordered_map<rvsdg::Input *, rvsdg::Output *> oldGammaNewGammaOutputMap;
  {
    for (const auto & [oldInput, oldBranchArgument] : oldGammaNode->GetEntryVars())
    {
      auto newGammaInput = oldGammaNewGammaInputMap[oldInput];
      auto newEntryVar =
          std::get<rvsdg::GammaNode::EntryVar>(newGammaNode->MapInput(*newGammaInput));
      auto newLoopVarInput = oldGammaNewThetaInputMap[oldInput];
      auto newLoopVar = newThetaNode->MapInputLoopVar(*newLoopVarInput);

      std::vector<rvsdg::Output *> values(2);
      values[exitSubregionIndex] = newEntryVar.branchArgument[exitSubregionIndex];
      values[repetitionSubregionIndex] = newLoopVar.output;
      auto newExitVar = newGammaNode->AddExitVar(values);
      oldGammaNewGammaOutputMap[oldInput] = newExitVar.output;
    }
  }

  // Stage 7 - Copy exit subregion into old theta node parent region
  rvsdg::SubstitutionMap stage7SMap;
  {
    for (const auto & [oldInput, oldBranchArgument] : oldGammaNode->GetEntryVars())
    {
      auto newOrigin = oldGammaNewGammaOutputMap[oldInput];
      stage7SMap.insert(oldBranchArgument[exitSubregionIndex], newOrigin);
    }

    oldExitSubregion->copy(oldThetaNode.region(), stage7SMap);
  }

  // Stage 8 - Replace old theta node outputs
  for (auto oldLoopVar : oldThetaNode.GetLoopVars())
  {
    auto oldExitVar = oldGammaNode->MapOutputExitVar(*oldLoopVar.post->origin());
    auto oldOrigin = oldExitVar.branchResult[exitSubregionIndex]->origin();
    auto & newOrigin = stage7SMap.lookup(*oldOrigin);
    oldLoopVar.output->divert_users(&newOrigin);
  }

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
    util::StatisticsCollector & statisticsCollector,
    std::shared_ptr<const LoopUnswitchingHeuristic> heuristic)
{
  LoopUnswitching loopUnswitching(std::move(heuristic));
  loopUnswitching.Run(rvsdgModule, statisticsCollector);
}

}

/*
 * Copyright 2025 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/opt/ScalarEvolution.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/util/Statistics.hpp>

#include <algorithm>
#include <queue>

namespace jlm::llvm
{
class ScalarEvolution::Statistics final : public util::Statistics
{

public:
  ~Statistics() noexcept override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Id::ScalarEvolution, sourceFile)
  {}

  void
  Start() noexcept
  {
    AddTimer(Label::Timer).start();
  }

  void
  Stop() noexcept
  {
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

ScalarEvolution::~ScalarEvolution() noexcept = default;

void
ScalarEvolution::Run(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFilePath().value());
  statistics->Start();

  InductionVariableMap_.clear();
  const rvsdg::Region & rootRegion = rvsdgModule.Rvsdg().GetRootRegion();
  TraverseRegion(rootRegion);

  statistics->Stop();
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
};

void
ScalarEvolution::TraverseRegion(const rvsdg::Region & region)
{
  for (const auto & node : region.Nodes())
  {
    if (const auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        TraverseRegion(subregion);
      }
      if (const auto thetaNode = dynamic_cast<const rvsdg::ThetaNode *>(structuralNode))
      {
        CreateChainRecurrences(*thetaNode);
      }
    }
  }
}

std::optional<const SCEV *>
ScalarEvolution::TryGetSCEVForOutput(const rvsdg::Output & output)
{
  if (const auto it = UniqueSCEVs_.find(&output); it != UniqueSCEVs_.end())
    return it->second.get();
  return std::nullopt;
}

std::unique_ptr<SCEV>
ScalarEvolution::GetOrCreateSCEVForOutput(const rvsdg::Output & output)
{
  if (const auto existing = TryGetSCEVForOutput(output))
  {
    return (*existing)->Clone();
  }

  std::unique_ptr<SCEV> result;
  if (rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    // We know this is a loop variable, create a placeholder SCEV for now, and compute the
    // expression later
    result = std::make_unique<SCEVPlaceholder>(output);
  }
  if (const auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
  {
    if (rvsdg::is<IntegerConstantOperation>(simpleNode->GetOperation()))
    {
      const auto constOp =
          dynamic_cast<const IntegerConstantOperation *>(&simpleNode->GetOperation());
      const auto value = constOp->Representation().to_uint();
      result = std::make_unique<SCEVConstant>(value);
    }
    if (rvsdg::is<IntegerAddOperation>(simpleNode->GetOperation()))
    {
      JLM_ASSERT(simpleNode->ninputs() == 2);
      const auto lhs = simpleNode->input(0)->origin();
      const auto rhs = simpleNode->input(1)->origin();
      result = std::make_unique<SCEVAddExpr>(
          GetOrCreateSCEVForOutput(*lhs),
          GetOrCreateSCEVForOutput(*rhs));
    }
  }
  // TODO: Handle more cases

  if (!result)
    // If none of the cases match, return an unknown SCEV expression
    result = std::make_unique<SCEVUnknown>();

  // Save the result in the cache
  UniqueSCEVs_[&output] = result->Clone();

  return result;
}

std::unordered_map<const rvsdg::Output *, int>
ScalarEvolution::FindDependenciesForSCEV(const SCEV & currentSCEV, const rvsdg::Output & currentIV)
{
  std::unordered_map<const rvsdg::Output *, int> dependencies{};
  if (dynamic_cast<const SCEVConstant *>(&currentSCEV)
      || dynamic_cast<const SCEVUnknown *>(&currentSCEV))
  {
    return dependencies;
  }
  if (const auto placeholderSCEV = dynamic_cast<const SCEVPlaceholder *>(&currentSCEV))
  {
    if (const auto dependency = placeholderSCEV->GetPrePointer())
      dependencies[dependency]++;
  }
  if (const auto addSCEV = dynamic_cast<const SCEVAddExpr *>(&currentSCEV))
  {
    // Recursively find dependencies on lhs and rhs
    std::unordered_map<const rvsdg::Output *, int> lhsDependencies =
        FindDependenciesForSCEV(*addSCEV->GetLeftOperand().get(), currentIV);
    std::unordered_map<const rvsdg::Output *, int> rhsDependencies =
        FindDependenciesForSCEV(*addSCEV->GetRightOperand().get(), currentIV);

    // Merge lhsDependencies into dependencies
    for (const auto & [ptr, count] : lhsDependencies)
      dependencies[ptr] += count;

    // Do the same for rhs
    for (const auto & [ptr, count] : rhsDependencies)
      dependencies[ptr] += count;
  }

  return dependencies;
}

ScalarEvolution::IVDependencyGraph
ScalarEvolution::CreateDependencyGraph(const rvsdg::ThetaNode & thetaNode) const
{
  IVDependencyGraph graph{};
  for (const auto loopVar : thetaNode.GetLoopVars())
  {
    const auto post = loopVar.post;
    const auto scev = UniqueSCEVs_.at(post->origin())->Clone();

    const auto pre = loopVar.pre;
    const std::unordered_map<const rvsdg::Output *, int> dependencies =
        FindDependenciesForSCEV(*scev.get(), *pre);
    graph[pre] = dependencies;
  }

  return graph;
}

// Implementation of Kahn's algorithm for topological sort
std::vector<const rvsdg::Output *>
ScalarEvolution::TopologicalSort(const IVDependencyGraph & dependencyGraph)
{
  const size_t numVertices = dependencyGraph.size();
  std::unordered_map<const rvsdg::Output *, int> indegree(numVertices);
  std::queue<const rvsdg::Output *> q{};
  for (const auto & [node, deps] : dependencyGraph)
  {
    for (const auto & dep : deps)
    {
      if (const auto ptr = dep.first; ptr == node)
        continue; // Ignore self-edges
      // To begin with, the indegree is just the number of incoming edges
      indegree[node] += 1;
    }
    if (indegree[node] == 0)
    {
      // Add nodes with no incoming edges to the queue, we know that these have no dependencies
      q.push(node);
    }
  }

  std::vector<const rvsdg::Output *> result{};
  while (!q.empty())
  {
    const rvsdg::Output * currentNode = q.front();
    q.pop();
    result.push_back(currentNode);

    for (const auto & [node, deps] : dependencyGraph)
    {
      if (node == currentNode)
        continue;

      for (const auto & dep : deps)
      {
        const auto ptr = dep.first;
        if (ptr == currentNode)
          continue;
        if (ptr == node)
        {
          // Update the indegree of nodes depending on this one
          indegree[node] -= 1;
          if (indegree[node] == 0)
            q.push(node);
        }
      }
    }
  }
  JLM_ASSERT(result.size() == numVertices && "Dependency graph can't contain cycles!");
  return result;
}

std::unique_ptr<SCEV>
ScalarEvolution::ReplacePlaceholders(
    const SCEV & scevTree,
    const rvsdg::Output & currentIV,
    const rvsdg::ThetaNode & thetaNode,
    const InductionVariableSet & validIVs)
{
  if (dynamic_cast<const SCEVConstant *>(&scevTree) || dynamic_cast<const SCEVUnknown *>(&scevTree))
  {
    return scevTree.Clone(); // Copy the leaf node
  }
  if (const auto placeholderSCEV = dynamic_cast<const SCEVPlaceholder *>(&scevTree))
  {
    const auto placeholderIV = placeholderSCEV->GetPrePointer();
    // Check if it is in the set of validIVs
    if (!validIVs.Contains(placeholderIV))
    {
      // If not, return an unknown
      return std::make_unique<SCEVUnknown>();
    }

    const auto loopVar = thetaNode.MapPreLoopVar(*placeholderIV);
    if (placeholderIV != &currentIV)
    {
      // We have a dependency of another IV
      // Get it's saved value. This is safe to do due to the topological ordering
      return UniqueSCEVs_.at(placeholderIV)->Clone();
    }
    // We have found ourselves, get the start value
    return GetOrCreateSCEVForOutput(*loopVar.input->origin())->Clone();
  }

  if (const auto addSCEV = dynamic_cast<const SCEVAddExpr *>(&scevTree))
  {
    auto left = ReplacePlaceholders(*addSCEV->GetLeftOperand(), currentIV, thetaNode, validIVs);
    auto right = ReplacePlaceholders(*addSCEV->GetRightOperand(), currentIV, thetaNode, validIVs);
    return std::make_unique<SCEVAddExpr>(std::move(left), std::move(right));
  }

  return scevTree.Clone();
}

bool
DependsOnLoopVariable(
    const rvsdg::Output & variable,
    ScalarEvolution::IVDependencyGraph & dependencyGraph)
{
  if (dependencyGraph[&variable].size() >= 1)
    return true;
  return false;
}

std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEV>>
ScalarEvolution::CreateChainRecurrences(const rvsdg::ThetaNode & thetaNode)
{
  for (const auto loopVar : thetaNode.GetLoopVars())
  {
    const auto post = loopVar.post;
    GetOrCreateSCEVForOutput(*post->origin());
  }
  auto dependencyGraph = CreateDependencyGraph(thetaNode);

  InductionVariableSet validIVs{};
  for (const auto loopVar : thetaNode.GetLoopVars())
  {
    if (!DependsOnLoopVariable(*loopVar.pre, dependencyGraph))
    {
      // If the expression doesn't depend on atleast one loop variable (including itself), it is not
      // an induction variable. Replace it with a SCEVUnknown
      UniqueSCEVs_.insert_or_assign(loopVar.post->origin(), std::make_unique<SCEVUnknown>());
    }
    else if (IsValidInductionVariable(*loopVar.pre, dependencyGraph))
      validIVs.insert(loopVar.pre);
  }

  // Filter the dependency graph to only contain the IVs that are valid and update dependencies
  // accordingly
  auto filteredDependencyGraph = dependencyGraph;
  for (auto it = filteredDependencyGraph.begin(); it != filteredDependencyGraph.end();)
  {
    if (!validIVs.Contains(it->first))
    {
      for (auto & [node, deps] : filteredDependencyGraph)
        deps.erase(it->first);
      it = filteredDependencyGraph.erase(it);
    }
    else
      ++it;
  }

  const auto order = TopologicalSort(filteredDependencyGraph);

  std::vector<const rvsdg::Output *> allVars{};
  // Add valid IVs to the set (in the correct order)
  for (const auto & indVarPre : order)
    allVars.push_back(indVarPre);
  for (const auto loopVar : thetaNode.GetLoopVars())
  {
    if (std::find(order.begin(), order.end(), loopVar.pre) == order.end())
      // This is not a valid IV, so it hasn't been added yet
      allVars.push_back(loopVar.pre);
  }

  for (const auto loopVarPre : allVars)
  {
    const auto loopVar = thetaNode.MapPreLoopVar(*loopVarPre);
    const auto post = loopVar.post;
    const auto SCEV = UniqueSCEVs_.at(post->origin()).get();
    auto replacedSCEV = ReplacePlaceholders(*SCEV, *loopVarPre, thetaNode, validIVs);
    UniqueSCEVs_.insert_or_assign(loopVarPre, std::move(replacedSCEV));
  }

  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEV>> chrecMap{};
  for (const auto loopVar : thetaNode.GetLoopVars())
    chrecMap[loopVar.pre] = UniqueSCEVs_.at(loopVar.pre)->Clone();

  return chrecMap;
}

bool
ScalarEvolution::IsValidInductionVariable(
    const rvsdg::Output & variable,
    IVDependencyGraph & dependencyGraph)
{
  // First check that variable has only one self-reference
  if (dependencyGraph[&variable][&variable] != 1)
    return false;

  // Then check for cycles through other variables
  std::unordered_set<const rvsdg::Output *> visited{};
  std::unordered_set<const rvsdg::Output *> recursionStack{};
  return !HasCycleThroughOthers(variable, dependencyGraph, visited, recursionStack);
}

bool
ScalarEvolution::HasCycleThroughOthers(
    const rvsdg::Output & currentIV,
    IVDependencyGraph & dependencyGraph,
    std::unordered_set<const rvsdg::Output *> & visited,
    std::unordered_set<const rvsdg::Output *> & recursionStack)
{
  visited.insert(&currentIV);
  recursionStack.insert(&currentIV);

  for (const auto & [depPtr, depCount] : dependencyGraph[&currentIV])
  {
    // Ignore self-references
    if (depPtr == &currentIV)
      continue;

    // Found a cycle back to a variable in our path
    if (recursionStack.find(depPtr) != recursionStack.end())
      return true;

    // Already explored this branch, no cycle
    if (visited.find(depPtr) != visited.end())
      continue;

    // Recursively check dependencies
    if (HasCycleThroughOthers(*depPtr, dependencyGraph, visited, recursionStack))
      return true;
  }

  recursionStack.erase(&currentIV);
  return false;
}

bool
ScalarEvolution::StructurallyEqual(const SCEV & a, const SCEV & b)
{
  if (typeid(a) != typeid(b))
    return false;

  if (auto * ca = dynamic_cast<const SCEVConstant *>(&a))
  {
    auto * cb = dynamic_cast<const SCEVConstant *>(&b);
    return ca->GetValue() == cb->GetValue();
  }

  if (dynamic_cast<const SCEVUnknown *>(&a))
    return true;

  if (auto * aa = dynamic_cast<const SCEVAddExpr *>(&a))
  {
    auto * ab = dynamic_cast<const SCEVAddExpr *>(&b);
    return StructurallyEqual(*aa->GetLeftOperand(), *ab->GetLeftOperand())
        && StructurallyEqual(*aa->GetRightOperand(), *ab->GetRightOperand());
  }

  if (auto * cha = dynamic_cast<const SCEVChrecExpr *>(&a))
  {
    auto * chb = dynamic_cast<const SCEVChrecExpr *>(&b);
    if (cha->GetLoop() != chb->GetLoop())
      return false;
    if (cha->Operands_.size() != chb->Operands_.size())
      return false;
    for (size_t i = 0; i < cha->Operands_.size(); ++i)
    {
      if (!StructurallyEqual(*cha->Operands_[i], *chb->Operands_[i]))
        return false;
    }
    return true;
  }

  return false;
}
}

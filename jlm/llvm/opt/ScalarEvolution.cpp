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
        const auto inductionVariables = FindInductionVariables(*thetaNode);
        InductionVariableMap_.emplace(thetaNode, inductionVariables);
        CreateChainRecurrences(inductionVariables, *thetaNode);
      }
    }
  }
}

bool
ScalarEvolution::IsBasedOnInductionVariable(
    const rvsdg::Output & output,
    InductionVariableSet & candidates)
{
  if (rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    // We know the output is a loop variable. Check if the loop variable is in the set, if so return
    // true, otherwise false
    return candidates.Contains(&output);
  }

  if (const auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
  {
    if (rvsdg::is<IntegerConstantOperation>(simpleNode->GetOperation()))
    {
      return true;
    }
    for (size_t n = 0; n < simpleNode->ninputs(); ++n)
    {
      const auto origin = simpleNode->input(n)->origin();
      if (!IsBasedOnInductionVariable(*origin, candidates))
      {
        return false;
      }
    }
    if (rvsdg::is<IntegerAddOperation>(simpleNode->GetOperation())
        || rvsdg::is<IntegerSubOperation>(simpleNode->GetOperation()))
    {
      return true;
    }
    return false;
  }
  // TODO: Handle structural nodes
  return false;
}

ScalarEvolution::InductionVariableSet
ScalarEvolution::FindInductionVariables(const rvsdg::ThetaNode & thetaNode)
{
  const std::vector<rvsdg::ThetaNode::LoopVar> loopVars = thetaNode.GetLoopVars();
  // Starting out, all loop variables are induction variable candidates
  InductionVariableSet inductionVariableCandidates{};
  for (const auto & loopVar : loopVars)
  {
    inductionVariableCandidates.Insert(loopVar.pre);
  }
  bool changed = false;
  do
  {
    for (const auto & loopVar : loopVars)
    {
      const rvsdg::Output * origin = loopVar.post->origin();
      if (!IsBasedOnInductionVariable(*origin, inductionVariableCandidates))
      {
        changed = inductionVariableCandidates.Remove(loopVar.pre);
      }
    }
  } while (changed == true);

  return inductionVariableCandidates;
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
      assert(simpleNode->ninputs() == 2);
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
    {
      if (dependencies.find(dependency) != dependencies.end())
        dependencies[dependency]++;
      else
        dependencies[dependency] = 1;
    }
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
    {
      dependencies[ptr] += count;
    }

    // Merge rhsDependencies into dependencies
    for (const auto & [ptr, count] : rhsDependencies)
    {
      dependencies[ptr] += count;
    }
  }

  return dependencies;
}

ScalarEvolution::IVDependencyGraph
ScalarEvolution::CreateDependencyGraph(
    const InductionVariableSet & inductionVariables,
    const rvsdg::ThetaNode & thetaNode) const
{
  IVDependencyGraph graph{};
  for (const auto indVarPre : inductionVariables.Items())
  {
    const auto loopVar = thetaNode.MapPreLoopVar(*indVarPre);
    const auto post = loopVar.post;
    const auto scev = UniqueSCEVs_.at(post->origin())->Clone();

    const std::unordered_map<const rvsdg::Output *, int> dependencies =
        FindDependenciesForSCEV(*scev.get(), *indVarPre);
    graph[indVarPre] = dependencies;
  }

  return graph;
}

// Implementation of Kahn's algorithm for topological sort
std::vector<const rvsdg::Output *>
ScalarEvolution::TopologicalSort(const IVDependencyGraph & dependencyGraph)
{
  const size_t numVertices = dependencyGraph.size();
  std::unordered_map<const rvsdg::Output *, int> indegree(numVertices);
  std::queue<const rvsdg::Output *> q;
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

  assert(result.size() == numVertices && "Dependency graph can't contain cycles!");
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

std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEV>>
ScalarEvolution::CreateChainRecurrences(
    const InductionVariableSet & inductionVariableCandidates,
    const rvsdg::ThetaNode & thetaNode)
{
  for (const auto indVarPre : inductionVariableCandidates.Items())
  {
    const auto loopVar = thetaNode.MapPreLoopVar(*indVarPre);
    const auto post = loopVar.post;
    GetOrCreateSCEVForOutput(*post->origin());
  }
  auto dependencyGraph = CreateDependencyGraph(inductionVariableCandidates, thetaNode);
  InductionVariableSet validIVs{ inductionVariableCandidates };
  for (const auto indVarPre : inductionVariableCandidates.Items())
  {
    if (!IsValidInductionVariable(*indVarPre, dependencyGraph))
      validIVs.Remove(indVarPre);
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

  const auto order{ TopologicalSort(filteredDependencyGraph) };

  std::vector<const rvsdg::Output *> allVars{};
  // Add valid IVs to the set (in the correct order)
  for (const auto & indVarPre : order)
    allVars.push_back(indVarPre);
  for (auto indVarPre : inductionVariableCandidates.Items())
  {
    if (std::find(order.begin(), order.end(), indVarPre) == order.end())
      // This is not a valid IV, so it hasn't been added yet
      allVars.push_back(indVarPre);
  }

  for (const auto indVarPre : allVars)
  {
    const auto loopVar = thetaNode.MapPreLoopVar(*indVarPre);
    const auto post = loopVar.post;
    const auto SCEV = UniqueSCEVs_.at(post->origin()).get();
    auto replacedSCEV = ReplacePlaceholders(*SCEV, *indVarPre, thetaNode, validIVs);
    UniqueSCEVs_.insert_or_assign(indVarPre, std::move(replacedSCEV));
  }

  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEV>> chrecMap{};
  for (auto indVar : inductionVariableCandidates.Items())
    chrecMap[indVar] = UniqueSCEVs_.at(indVar)->Clone();

  return chrecMap;
}

bool
ScalarEvolution::IsValidInductionVariable(
    const rvsdg::Output & variable,
    IVDependencyGraph & dependencyGraph)
{
  // First check for multiple self-references
  for (const auto & [depPtr, depCount] : dependencyGraph[&variable])
  {
    if (depPtr == &variable && depCount > 1)
      return false;
  }

  // Then check for cycles through other variables
  std::unordered_set<const rvsdg::Output *> visited;
  std::unordered_set<const rvsdg::Output *> recursionStack;
  return !HasCycleThroughOthers(&variable, dependencyGraph, visited, recursionStack);
}

bool
ScalarEvolution::HasCycleThroughOthers(
    const rvsdg::Output * current,
    IVDependencyGraph & dependencyGraph,
    std::unordered_set<const rvsdg::Output *> & visited,
    std::unordered_set<const rvsdg::Output *> & recursionStack)
{
  visited.insert(current);
  recursionStack.insert(current);

  for (const auto & [depPtr, depCount] : dependencyGraph[current])
  {
    // Ignore sel-references
    if (depPtr == current)
      continue;

    // Found a cycle back to a variable in our path
    if (recursionStack.find(depPtr) != recursionStack.end())
      return true;

    // Already explored this branch, no cycle
    if (visited.find(depPtr) != visited.end())
      continue;

    // Recursively check dependencies
    if (HasCycleThroughOthers(depPtr, dependencyGraph, visited, recursionStack))
      return true;
  }

  recursionStack.erase(current);
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

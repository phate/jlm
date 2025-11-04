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
        PerformSCEVAnalysis(*thetaNode);
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
        FindDependenciesForSCEV(*addSCEV->GetLeftOperand(), currentIV);
    std::unordered_map<const rvsdg::Output *, int> rhsDependencies =
        FindDependenciesForSCEV(*addSCEV->GetRightOperand(), currentIV);

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
ScalarEvolution::PerformSCEVAnalysis(const rvsdg::ThetaNode & thetaNode)
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
    if (std::find(order.begin(), order.end(), loopVarPre) != order.end())
    {
      const auto loopVar = thetaNode.MapPreLoopVar(*loopVarPre);
      const auto post = loopVar.post;
      const auto scev = UniqueSCEVs_.at(post->origin()).get();

      auto chainRecurrence = CreateChainRecurrence(*loopVarPre, *scev, thetaNode);

      if (auto simpleNode =
              rvsdg::TryGetOwnerNode<const rvsdg::SimpleNode>(*loopVar.input->origin()))
      {
        if (rvsdg::is<IntegerConstantOperation>(simpleNode->GetOperation()))
        {
          // If the input value is a constant, get it's SCEV representation
          chainRecurrence->SetStartValue(
              GetOrCreateSCEVForOutput(*loopVar.input->origin())->Clone());
        }
      }
      else
      {
        // If not, create a SCEVInit node representing the start value
        chainRecurrence->SetStartValue(std::make_unique<SCEVInit>(*loopVarPre));
      }

      ChainRecurrenceMap_.emplace(loopVarPre, std::move(chainRecurrence));
    }
    else
    {
      auto unknownChainRecurrence = std::make_unique<SCEVChainRecurrence>(thetaNode);
      unknownChainRecurrence->AddOperand(std::make_unique<SCEVUnknown>());
      ChainRecurrenceMap_.emplace(loopVarPre, std::move(unknownChainRecurrence));
    }

    std::cout << loopVarPre->debug_string() << ": "
              << ChainRecurrenceMap_.at(loopVarPre)->DebugString() << '\n';
  }

  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEV>> scevMap{};
  for (const auto loopVarPre : order)
    scevMap[loopVarPre] = UniqueSCEVs_.at(loopVarPre)->Clone();

  return scevMap;
}

std::unique_ptr<SCEVChainRecurrence>
ScalarEvolution::CreateChainRecurrence(
    const rvsdg::Output & IV,
    const SCEV & scevTree,
    const rvsdg::ThetaNode & thetaNode)
{
  auto chrec = std::make_unique<SCEVChainRecurrence>(thetaNode);

  if (const auto scevConstant = dynamic_cast<const SCEVConstant *>(&scevTree))
  {
    // This is a constant, we add it as the only operand
    chrec->AddOperand(scevConstant->Clone());
  }
  else if (const auto scevPlaceholder = dynamic_cast<const SCEVPlaceholder *>(&scevTree))
  {
    if (scevPlaceholder->GetPrePointer() == &IV)
    {
      // Since we are only interested in the step value, and not the initial value, we can ignore
      // ourselves by adding a 0, which is the identity element
      chrec->AddOperand(std::make_unique<SCEVConstant>(0));
    }
    else
    {
      if (ChainRecurrenceMap_.find(scevPlaceholder->GetPrePointer()) != ChainRecurrenceMap_.end())
      {
        // We have a dependency of another IV
        // Get it's saved value. This is safe to do due to the topological ordering
        auto storedRec = ChainRecurrenceMap_.at(scevPlaceholder->GetPrePointer())->Clone();

        return std::unique_ptr<SCEVChainRecurrence>(
            dynamic_cast<SCEVChainRecurrence *>(storedRec.release()));
      }
      chrec->AddOperand(std::make_unique<SCEVUnknown>());
    }
  }
  else if (const auto scevAddExpr = dynamic_cast<const SCEVAddExpr *>(&scevTree))
  {
    auto lhsChrec = CreateChainRecurrence(IV, *scevAddExpr->GetLeftOperand(), thetaNode);
    auto rhsChrec = CreateChainRecurrence(IV, *scevAddExpr->GetRightOperand(), thetaNode);
    std::cout << "Folding : " << lhsChrec->DebugString() << " and " << rhsChrec->DebugString()
              << " into: ";

    /* Apply folding rules
     *
     * We have the following folding rules from the CR algebra:
     * G + {e,+,f}         =>       {G + e,+,f}         (1)
     * {e,+,f} + {g,+,h}   =>       {e + g,+,f + h}     (2)
     *
     * The loop below is able to apply both folding rules at the same time.
     * We do this with a generalized version of rule (2), where for two recurrences of arbitrary
     * length, we add the operands element by element. If the recurrences being added together are
     * not the same size, we pad the shorter recurrence with 0's.
     *
     * It is clear that these statements are equivalent:
     * {G,+,0} + {e,+,f} = {G + e,+,0 + f} = {G + e,+,f}
     *
     * Since we represent constants in the SCEVTree as recurrences consisting of only a SCEVConstant
     * node, we can therefore pad the constant recurrence with however many zeroes we need for the
     * length of the other recurrence. This effectively let's us apply both rules in one go.
     */

    const auto lhsSize = lhsChrec->GetOperands().size();
    const auto rhsSize = rhsChrec->GetOperands().size();
    for (size_t i = 0; i < std::max(lhsSize, rhsSize); ++i)
    {
      SCEV * lhsOperand{};
      SCEV * rhsOperand{};
      if (i <= lhsSize - 1)
        lhsOperand = lhsChrec->GetOperand(i);

      if (i <= rhsSize - 1)
        rhsOperand = rhsChrec->GetOperand(i);

      if (dynamic_cast<SCEVUnknown *>(lhsOperand) || dynamic_cast<SCEVUnknown *>(rhsOperand))
      {
        chrec->AddOperand(std::make_unique<SCEVUnknown>());
        break;
      }

      // Set default values to 0, that way, we can effectively pad the recurrence, if no element is
      // found for the current index
      uint64_t lhsValue = 0;
      uint64_t rhsValue = 0;
      if (lhsOperand)
      {
        if (auto lhsConstant = dynamic_cast<SCEVConstant *>(lhsOperand))
          lhsValue = lhsConstant->GetValue();
      }
      if (rhsOperand)
      {
        if (auto rhsConstant = dynamic_cast<SCEVConstant *>(rhsOperand))
          rhsValue = rhsConstant->GetValue();
      }

      if (lhsValue + rhsValue != 0)
        chrec->AddOperand(std::make_unique<SCEVConstant>(lhsValue + rhsValue));
    }
    std::cout << chrec->DebugString() << '\n';
  }
  else
  {
    // Unknown SCEV type - add a fallback
    std::cerr << "Warning: Unknown SCEV type in CreateChainRecurrence\n";
    chrec->AddOperand(std::make_unique<SCEVUnknown>());
  }

  return chrec;
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

  if (auto * cha = dynamic_cast<const SCEVChainRecurrence *>(&a))
  {
    auto * chb = dynamic_cast<const SCEVChainRecurrence *>(&b);
    if (cha->GetLoop() != chb->GetLoop())
      return false;
    if (cha->GetOperands().size() != chb->GetOperands().size())
      return false;
    for (size_t i = 0; i < cha->GetOperands().size(); ++i)
    {
      if (!StructurallyEqual(*cha->GetOperands()[i], *chb->GetOperands()[i]))
        return false;
    }
    return true;
  }

  return false;
}
}

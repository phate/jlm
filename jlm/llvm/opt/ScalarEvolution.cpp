/*
 * Copyright 2025 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/ir/types.hpp>
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
  Stop(const Context & context) noexcept
  {
    GetTimer(Label::Timer).stop();
    AddMeasurement(Label::NumTotalRecurrences, context.GetNumOfTotalChrecs());
    AddMeasurement(Label::NumConstantRecurrences, context.GetNumOfChrecsWithOrder(0));
    AddMeasurement(Label::NumFirstOrderRecurrences, context.GetNumOfChrecsWithOrder(1));
    AddMeasurement(Label::NumSecondOrderRecurrences, context.GetNumOfChrecsWithOrder(2));
    AddMeasurement(Label::NumThirdOrderRecurrences, context.GetNumOfChrecsWithOrder(3));
    AddMeasurement(Label::NumLoopVariablesTotal, context.GetNumTotalLoopVars());
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

template<typename T>
std::unique_ptr<T>
clone_as(const SCEV & scev)
{
  auto cloned = scev.Clone();
  auto * ptr = dynamic_cast<T *>(cloned.release());
  JLM_ASSERT(ptr && "Cannot create unique_ptr from pointer because pointer is undefined!");
  return std::unique_ptr<T>(ptr);
}

std::unique_ptr<SCEVChainRecurrence>
ScalarEvolution::Context::TryGetChrecForOutput(const rvsdg::Output & output) const
{
  const auto it = ChrecMap_.find(&output);
  if (it == ChrecMap_.end() || !it->second)
    return nullptr;

  return clone_as<SCEVChainRecurrence>(*it->second);
}

std::unique_ptr<SCEV>
ScalarEvolution::Context::TryGetSCEVForOutput(const rvsdg::Output & output) const
{
  const auto it = SCEVMap_.find(&output);
  if (it == SCEVMap_.end() || !it->second)
    return nullptr;

  return it->second->Clone();
}

void
ScalarEvolution::Context::InsertChrec(
    const rvsdg::Output & output,
    const std::unique_ptr<SCEVChainRecurrence> & chrec)
{
  ChrecMap_.insert_or_assign(&output, clone_as<SCEVChainRecurrence>(*chrec));
}

std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEVChainRecurrence>>
ScalarEvolution::Context::GetChrecs() const
{
  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEVChainRecurrence>> mapCopy{};
  for (auto & [output, chrec] : ChrecMap_)
  {
    mapCopy.emplace(output, clone_as<SCEVChainRecurrence>(*chrec));
  }
  return mapCopy;
}

int
ScalarEvolution::Context::GetNumOfChrecsWithOrder(const int n) const
{
  int count = 0;
  for (auto & [out, chrec] : ChrecMap_)
  {
    // Count chrecs with specific order
    if (static_cast<int>(chrec->GetOperands().size()) == n + 1)
      count++;
  }
  return count;
}

size_t
ScalarEvolution::Context::GetNumOfTotalChrecs() const
{
  return ChrecMap_.size();
}

void
ScalarEvolution::Context::AddLoopVar(const rvsdg::Output & var)
{
  LoopVars_.push_back(&var);
}

size_t
ScalarEvolution::Context::GetNumTotalLoopVars() const
{
  return LoopVars_.size();
}

void
ScalarEvolution::Context::InsertSCEV(
    const rvsdg::Output & output,
    const std::unique_ptr<SCEV> & scev)
{
  SCEVMap_.insert_or_assign(&output, scev->Clone());
}

ScalarEvolution::ScalarEvolution()
    : Transformation("ScalarEvolution")
{}

ScalarEvolution::~ScalarEvolution() noexcept = default;

void
ScalarEvolution::Run(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFilePath().value());
  statistics->Start();

  const auto ctx = Context::Create();
  const rvsdg::Region & rootRegion = rvsdgModule.Rvsdg().GetRootRegion();
  AnalyzeRegion(rootRegion, *ctx);

  statistics->Stop(*ctx);
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
};

bool
ScalarEvolution::IsUnknown(const SCEVChainRecurrence & chrec)
{
  for (const auto operand : chrec.GetOperands())
  {
    if (dynamic_cast<const SCEVUnknown *>(operand))
    {
      return true;
    }
  }
  return false;
}

void
ScalarEvolution::AnalyzeRegion(const rvsdg::Region & region, Context & ctx)
{
  for (const auto & node : region.Nodes())
  {
    if (const auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        AnalyzeRegion(subregion, ctx);
      }
      if (const auto thetaNode = dynamic_cast<const rvsdg::ThetaNode *>(structuralNode))
      {
        // Add number of loop vars in theta (for statistics)
        for (const auto loopVar : thetaNode->GetLoopVars())
        {
          if (loopVar.pre->Type()->Kind() != rvsdg::TypeKind::State)
          {
            // Only add loop variables that are not states
            ctx.AddLoopVar(*loopVar.pre);
          }
        }

        PerformSCEVAnalysis(*thetaNode, ctx);
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
ScalarEvolution::GetNegativeSCEV(const SCEV & scev)
{
  // -(c)
  if (const auto c = dynamic_cast<const SCEVConstant *>(&scev))
  {
    const auto value = c->GetValue();
    return std::make_unique<SCEVConstant>(-value);
  }
  // -(-x) -> x
  if (const auto mul = dynamic_cast<const SCEVMulExpr *>(&scev))
  {
    if (const auto c = dynamic_cast<const SCEVConstant *>(mul->GetLeftOperand());
        c && c->GetValue() == -1)
    {
      return mul->GetRightOperand()->Clone();
    }
    if (const auto c = dynamic_cast<const SCEVConstant *>(mul->GetRightOperand());
        c && c->GetValue() == -1)
    {
      return mul->GetLeftOperand()->Clone();
    }
  } // -(x + y) -> (-x) + (-y)
  if (const auto add = dynamic_cast<const SCEVAddExpr *>(&scev))
  {
    return std::make_unique<SCEVAddExpr>(
        GetNegativeSCEV(*add->GetLeftOperand()),
        GetNegativeSCEV(*add->GetRightOperand()));
  }
  // General case: -(x) -> (-1) * x
  return std::make_unique<SCEVMulExpr>(std::make_unique<SCEVConstant>(-1), scev.Clone());
}

std::unique_ptr<SCEV>
ScalarEvolution::GetOrCreateSCEVForOutput(const rvsdg::Output & output, Context & ctx)
{
  if (const auto existing = ctx.TryGetSCEVForOutput(output))
    return existing->Clone();

  std::unique_ptr<SCEV> result{};
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
      const auto value = constOp->Representation().to_int();
      result = std::make_unique<SCEVConstant>(value);
    }
    if (rvsdg::is<IntegerAddOperation>(simpleNode->GetOperation()))
    {
      JLM_ASSERT(simpleNode->ninputs() == 2);
      const auto lhs = simpleNode->input(0)->origin();
      const auto rhs = simpleNode->input(1)->origin();

      auto lhsScev = GetOrCreateSCEVForOutput(*lhs, ctx);
      auto rhsScev = GetOrCreateSCEVForOutput(*rhs, ctx);

      result = std::make_unique<SCEVAddExpr>(std::move(lhsScev), std::move(rhsScev));
    }
    if (rvsdg::is<IntegerSubOperation>(simpleNode->GetOperation()))
    {
      JLM_ASSERT(simpleNode->ninputs() == 2);
      const auto lhs = simpleNode->input(0)->origin();
      const auto rhs = simpleNode->input(1)->origin();

      auto lhsScev = GetOrCreateSCEVForOutput(*lhs, ctx);
      auto rhsScev = GetOrCreateSCEVForOutput(*rhs, ctx);

      auto rhsNegativeScev = GetNegativeSCEV(*rhsScev);

      result = std::make_unique<SCEVAddExpr>(std::move(lhsScev), std::move(rhsNegativeScev));
    }
    if (rvsdg::is<IntegerMulOperation>(simpleNode->GetOperation()))
    {
      JLM_ASSERT(simpleNode->ninputs() == 2);
      const auto lhs = simpleNode->input(0)->origin();
      const auto rhs = simpleNode->input(1)->origin();

      auto lhsScev = GetOrCreateSCEVForOutput(*lhs, ctx);
      auto rhsScev = GetOrCreateSCEVForOutput(*rhs, ctx);

      result = std::make_unique<SCEVMulExpr>(std::move(lhsScev), std::move(rhsScev));
    }
  }

  if (!result)
    // If none of the cases match, return an unknown SCEV expression
    result = std::make_unique<SCEVUnknown>();

  // Save the result in the cache
  ctx.InsertSCEV(output, result);

  return result;
}

void
ScalarEvolution::FindDependenciesForSCEV(
    const SCEV & scev,
    DependencyMap & dependencies,
    const DependencyOp op = DependencyOp::None)
{
  if (const auto placeholderSCEV = dynamic_cast<const SCEVPlaceholder *>(&scev))
  {
    if (const auto dependency = placeholderSCEV->GetPrePointer())
    {
      // Retrieves dependency info struct from the map
      // In the case where the dependency does not already exist, a new struct is created with the
      // default count being 0 and the default operation being None
      auto & depInfo = dependencies[dependency];
      depInfo.operation = op;
      depInfo.count++;
    }
  }

  if (const auto addSCEV = dynamic_cast<const SCEVAddExpr *>(&scev))
  {
    FindDependenciesForSCEV(*addSCEV->GetLeftOperand(), dependencies, DependencyOp::Add);
    FindDependenciesForSCEV(*addSCEV->GetRightOperand(), dependencies, DependencyOp::Add);
  }

  if (const auto mulSCEV = dynamic_cast<const SCEVMulExpr *>(&scev))
  {
    // Only pass Mul down if we haven't already seen Add in the path from root
    // If op is already Add, preserve it; otherwise use Mul
    const DependencyOp opToPass = (op == DependencyOp::Add) ? DependencyOp::Add : DependencyOp::Mul;
    FindDependenciesForSCEV(*mulSCEV->GetLeftOperand(), dependencies, opToPass);
    FindDependenciesForSCEV(*mulSCEV->GetRightOperand(), dependencies, opToPass);
  }
}

ScalarEvolution::IVDependencyGraph
ScalarEvolution::CreateDependencyGraph(const rvsdg::ThetaNode & thetaNode, const Context & ctx)
{
  IVDependencyGraph graph{};
  for (const auto loopVar : thetaNode.GetLoopVars())
  {
    const auto post = loopVar.post;
    const auto scev = ctx.TryGetSCEVForOutput(*post->origin());

    DependencyMap dependencies{};
    FindDependenciesForSCEV(*scev.get(), dependencies);

    const auto pre = loopVar.pre;
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
        if (ptr == node)
          continue; // Skip self-edges
        if (ptr == currentNode)
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

void
ScalarEvolution::PerformSCEVAnalysis(const rvsdg::ThetaNode & thetaNode, Context & ctx)
{
  for (const auto loopVar : thetaNode.GetLoopVars())
  {
    const auto post = loopVar.post;
    GetOrCreateSCEVForOutput(*post->origin(), ctx);
  }
  auto dependencyGraph = CreateDependencyGraph(thetaNode, ctx);

  util::HashSet<const rvsdg::Output *> validIVs{};
  for (const auto loopVar : thetaNode.GetLoopVars())
  {
    if (!DependsOnLoopVariable(*loopVar.pre, dependencyGraph))
    {
      // If the expression doesn't depend on at least one loop variable (including itself), it is
      // not an induction variable. Replace it with a SCEVUnknown
      ctx.InsertSCEV(*loopVar.post->origin(), std::make_unique<SCEVUnknown>());
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

  for (size_t i = 0; i < order.size(); ++i)
  {
    // Process valid induction variables
    const auto loopVarPre = allVars[i];
    const auto loopVar = thetaNode.MapPreLoopVar(*loopVarPre);
    const auto post = loopVar.post;
    const auto scev = ctx.TryGetSCEVForOutput(*post->origin());

    auto chainRecurrence = GetOrCreateChainRecurrence(*loopVarPre, *scev, thetaNode, ctx);

    if (chainRecurrence->GetOperands().size() == 1
        && StructurallyEqual(*chainRecurrence->GetOperand(0), SCEVConstant(0)))
    {
      // If the recurrence is empty ({0}), delete the old unique_ptr and create a new one
      // without any operands. This effectively removes trailing zeroes for constants
      chainRecurrence.reset();
      chainRecurrence = std::make_unique<SCEVChainRecurrence>(thetaNode);
    }

    // Find the start value for the recurrence
    if (auto const constantInteger = tryGetConstantSignedInteger(*loopVar.input->origin()))
    {
      // If the input value is a constant, create a SCEV representation and set it as start
      // value (first operand in rec)
      chainRecurrence->AddOperandToFront(std::make_unique<SCEVConstant>(*constantInteger));
    }
    else
    {
      // If not, create a SCEVInit node representing the start value
      chainRecurrence->AddOperandToFront(std::make_unique<SCEVInit>(*loopVarPre));
    }
    ctx.InsertChrec(*loopVarPre, chainRecurrence);
  }

  for (size_t i = order.size(); i < allVars.size(); ++i)
  {
    // Handle invalid induction variables
    const auto loopVarPre = allVars[i];
    auto unknownChainRecurrence = std::make_unique<SCEVChainRecurrence>(thetaNode);
    unknownChainRecurrence->AddOperand(std::make_unique<SCEVUnknown>());
    ctx.InsertChrec(*loopVarPre, unknownChainRecurrence);
  }
}

std::unique_ptr<SCEVChainRecurrence>
ScalarEvolution::GetOrCreateChainRecurrence(
    const rvsdg::Output & output,
    const SCEV & scevTree,
    const rvsdg::ThetaNode & thetaNode)
{
  if (const auto existing = ctx.TryGetChrecForOutput(output))
  {
    return clone_as<SCEVChainRecurrence>(*existing);
  }

  auto chrec = std::make_unique<SCEVChainRecurrence>(thetaNode);

  if (const auto scevConstant = dynamic_cast<const SCEVConstant *>(&scevTree))
  {
    // This is a constant, we add it as the only operand
    chrec->AddOperand(scevConstant->Clone());
    return chrec;
  }
  if (const auto scevPlaceholder = dynamic_cast<const SCEVPlaceholder *>(&scevTree))
  {
    if (scevPlaceholder->GetPrePointer() == &output)
    {
      // Since we are only interested in the step value, and not the initial value, we can ignore
      // ourselves by returning an empty chain recurrence (treated as the identity element - 0 for
      // addition and 1 for multiplication)
      return chrec;
    }
    if (auto storedRec = ctx.TryGetChrecForOutput(*scevPlaceholder->GetPrePointer()))
    {
      // We have a dependency of another IV
      // Get it's saved value. This is safe to do due to the topological ordering
      return storedRec;
    }
    chrec->AddOperand(std::make_unique<SCEVUnknown>());
    return chrec;
  }
  if (const auto scevAddExpr = dynamic_cast<const SCEVAddExpr *>(&scevTree))
  {
    // We have the following folding rules from the CR algebra:
    // G + {e,+,f}         =>       {G + e,+,f}         (1)
    // {e,+,f} + {g,+,h}   =>       {e + g,+,f + h}     (2)
    //
    // And by generalizing rule 2, we have that:
    // {G,+,0} + {e,+,f} = {G + e,+,0 + f} = {G + e,+,f}
    //
    // Since we represent constants in the SCEVTree as recurrences consisting of only a SCEVConstant
    // node, we can therefore pad the constant recurrence with however many zeroes we need for the
    // length of the other recurrence. This effectively lets us apply both rules in one go.

    const auto lhsChrec =
        GetOrCreateChainRecurrence(output, *scevAddExpr->GetLeftOperand(), thetaNode, ctx);
    const auto rhsChrec =
        GetOrCreateChainRecurrence(output, *scevAddExpr->GetRightOperand(), thetaNode, ctx);

    const auto lhsSize = lhsChrec->GetOperands().size();
    const auto rhsSize = rhsChrec->GetOperands().size();
    for (size_t i = 0; i < std::max(lhsSize, rhsSize); ++i)
    {
      const SCEV * lhsOperand{};
      const SCEV * rhsOperand{};
      if (i < lhsSize)
        lhsOperand = lhsChrec->GetOperand(i);

      if (i < rhsSize)
        rhsOperand = rhsChrec->GetOperand(i);
      chrec->AddOperand(ApplyAddFolding(lhsOperand, rhsOperand));
    }
    return chrec;
  }
  if (const auto scevMulExpr = dynamic_cast<const SCEVMulExpr *>(&scevTree))
  {
    // We have the following folding rules from the CR algebra:
    // G * {e,+,f}         =>       {G * e,+,G * f}
    // {e,+,f} * {g,+,h}   =>       {e * g,+,e * h + f * g + f * h,+,2*f*h}
    //
    // AFAIK these are the only rules that we are able to support as there is no general rule for
    // folding two addrecs of arbitrary length with the multiplication operation

    const auto lhsChrec =
        GetOrCreateChainRecurrence(output, *scevMulExpr->GetLeftOperand(), thetaNode, ctx);
    const auto rhsChrec =
        GetOrCreateChainRecurrence(output, *scevMulExpr->GetRightOperand(), thetaNode, ctx);

      // Third step: 2 * f * h
      chrec->AddOperand(ApplyMulFolding(std::make_unique<SCEVConstant>(2).get(), fh.get()));
    }
    else
    {
      // For other cases, return unknown
      chrec->AddOperand(std::make_unique<SCEVUnknown>());
    }
    return chrec;
  }
  JLM_ASSERT(false && "Unknown SCEV type in CreateChainRecurrence!");
  return chrec;
}

bool
isNonZeroConstant(const SCEVConstant * c)
{
  return c && c->GetValue() != 0;
}

std::unique_ptr<SCEV>
ScalarEvolution::ApplyAddFolding(const SCEV * lhsOperand, const SCEV * rhsOperand)
{
  // Apply folding rules for addition
  //
  // For constants and unknowns this is trivial, however it becomes a bit complicated when we
  // factor in SCEVInit nodes. These nodes represent the initial value of an IV in the case where
  // the exact value is unknown at compile time. E.g. function argument or result from a
  // call-instruction. In the cases where we have to fold one or more of these init-nodes, we create
  // an n-ary add expression (add expression with an arbitrary number of operands), and add this to
  // the chrec. Folding two of these n-ary add expressions will result in another n-ary add
  // expression, which consists of all the operands in both the left and the right expression.

  // The if-chain below goes through each of the possible combinations of lhs and rhs values
  if (const auto *lhsUnknown = dynamic_cast<const SCEVUnknown *>(lhsOperand),
      *rhsUnknown = dynamic_cast<const SCEVUnknown *>(rhsOperand);
      lhsUnknown || rhsUnknown)
  {
    // If one of the sides is unknown. Return unknown
    return std::make_unique<SCEVUnknown>();
  }

  const auto lhsNAryMulExpr = dynamic_cast<const SCEVNAryMulExpr *>(lhsOperand);
  const auto rhsNAryMulExpr = dynamic_cast<const SCEVNAryMulExpr *>(rhsOperand);
  // Handle n-ary multiply expressions - they become terms in an n-ary add expression
  if (lhsNAryMulExpr && rhsNAryMulExpr)
  {
    // Two multiply expressions - create add expression with both
    return std::make_unique<SCEVNAryAddExpr>(lhsNAryMulExpr->Clone(), rhsNAryMulExpr->Clone());
  }

  const auto lhsNAryAddExpr = dynamic_cast<const SCEVNAryAddExpr *>(lhsOperand);
  const auto rhsNAryAddExpr = dynamic_cast<const SCEVNAryAddExpr *>(rhsOperand);
  if ((lhsNAryMulExpr && rhsNAryAddExpr) || (rhsNAryMulExpr && lhsNAryAddExpr))
  {
    // Multiply expression with add expression - Clone the add expression and add the multiply as a
    // term
    const auto * mulExpr = lhsNAryMulExpr ? lhsNAryMulExpr : rhsNAryMulExpr;
    auto * addExpr = lhsNAryAddExpr ? lhsNAryAddExpr : rhsNAryAddExpr;
    auto addExprClone = addExpr->Clone();
    auto newAddExpr = dynamic_cast<SCEVNAryAddExpr *>(addExprClone.get());
    newAddExpr->AddOperand(mulExpr->Clone());
    return newAddExpr->Clone();
  }

  const auto lhsInit = dynamic_cast<const SCEVInit *>(lhsOperand);
  const auto rhsInit = dynamic_cast<const SCEVInit *>(rhsOperand);
  if ((lhsNAryMulExpr && rhsInit) || (rhsNAryMulExpr && lhsInit))
  {
    // Multiply expression with init - create add expression
    const auto * mulExpr = lhsNAryMulExpr ? lhsNAryMulExpr : rhsNAryMulExpr;
    const auto * init = lhsInit ? lhsInit : rhsInit;
    return std::make_unique<SCEVNAryAddExpr>(mulExpr->Clone(), init->Clone());
  }

  const auto lhsConstant = dynamic_cast<const SCEVConstant *>(lhsOperand);
  const auto rhsConstant = dynamic_cast<const SCEVConstant *>(rhsOperand);
  if ((lhsNAryMulExpr && isNonZeroConstant(rhsConstant))
      || (rhsNAryMulExpr && isNonZeroConstant(lhsConstant)))
  {
    // Multiply expression with nonzero constant - create add expression
    const auto * mulExpr = lhsNAryMulExpr ? lhsNAryMulExpr : rhsNAryMulExpr;
    const auto * constant = lhsConstant ? lhsConstant : rhsConstant;
    return std::make_unique<SCEVNAryAddExpr>(mulExpr->Clone(), constant->Clone());
  }

  if (lhsNAryMulExpr || rhsNAryMulExpr)
  {
    // Single multiply expression, no folding necessary
    const auto * mulExpr = lhsNAryMulExpr ? lhsNAryMulExpr : rhsNAryMulExpr;
    return mulExpr->Clone();
  }

  if (lhsInit && rhsInit)
  {
    // We have two init nodes. Create a nAryAdd with lhsInit and rhsInit
    return std::make_unique<SCEVNAryAddExpr>(lhsInit->Clone(), rhsInit->Clone());
  }

  if ((lhsInit && rhsNAryAddExpr) || (rhsInit && lhsNAryAddExpr))
  {
    // We have an init and an add expr. Clone the add expression and add the init as an operand
    const auto * init = lhsInit ? lhsInit : rhsInit;
    auto * nAryAddExpr = lhsNAryAddExpr ? lhsNAryAddExpr : rhsNAryAddExpr;
    auto addExprClone = nAryAddExpr->Clone();
    auto newAddExpr = dynamic_cast<SCEVNAryAddExpr *>(addExprClone.get());
    newAddExpr->AddOperand(init->Clone());
    return newAddExpr->Clone();
  }

  if ((lhsInit && isNonZeroConstant(rhsConstant)) || (rhsInit && isNonZeroConstant(lhsConstant)))
  {
    // We have an init and a nonzero constant. Create a nAryAdd with init and constant
    const auto * init = lhsInit ? lhsInit : rhsInit;
    const auto * constant = lhsConstant ? lhsConstant : rhsConstant;
    return std::make_unique<SCEVNAryAddExpr>(init->Clone(), constant->Clone());
  }

  if (lhsInit || rhsInit)
  {
    // Only one operand. Add it
    const auto * init = lhsInit ? lhsInit : rhsInit;
    return init->Clone();
  }

  if (lhsNAryAddExpr && rhsNAryAddExpr)
  {
    // We have two add expressions. Clone the lhs and add the rhs operands
    auto lhsNAryAddExprClone = lhsNAryAddExpr->Clone();
    auto lhsNewNAryAddExpr = dynamic_cast<SCEVNAryAddExpr *>(lhsNAryAddExprClone.get());
    for (auto op : rhsNAryAddExpr->GetOperands())
    {
      lhsNewNAryAddExpr->AddOperand(op->Clone());
    }
    return lhsNewNAryAddExpr->Clone();
  }

  if ((lhsNAryAddExpr && isNonZeroConstant(rhsConstant))
      || (rhsNAryAddExpr && isNonZeroConstant(lhsConstant)))
  {
    // We have an add expr and a nonzero constant. Clone the add expr and add the constant
    auto * nAryAddExpr = lhsNAryAddExpr ? lhsNAryAddExpr : rhsNAryAddExpr;
    auto * constant = lhsConstant ? lhsConstant : rhsConstant;
    auto nAryAddExprClone = nAryAddExpr->Clone();
    auto newNAryAddExpr = dynamic_cast<SCEVNAryAddExpr *>(nAryAddExprClone.get());
    newNAryAddExpr->AddOperand(constant->Clone());
    return newNAryAddExpr->Clone();
  }

  if (lhsNAryAddExpr || rhsNAryAddExpr)
  {
    const auto * nAryAddExpr = lhsNAryAddExpr ? lhsNAryAddExpr : rhsNAryAddExpr;
    return nAryAddExpr->Clone();
  }
  if (lhsConstant && rhsConstant)
  {
    // Two constants, get their value, and combine them (fold)
    const auto lhsValue = lhsConstant->GetValue();
    const auto rhsValue = rhsConstant->GetValue();

    return std::make_unique<SCEVConstant>(lhsValue + rhsValue);
  }

  if (lhsConstant || rhsConstant)
  {
    const auto * constant = lhsConstant ? lhsConstant : rhsConstant;
    return constant->Clone();
  }

  return std::make_unique<SCEVUnknown>();
}

std::unique_ptr<SCEV>
ScalarEvolution::ApplyMulFolding(const SCEV * lhsOperand, const SCEV * rhsOperand)
{
  // Apply folding rules for multiplication
  //
  // Similar to addition, we need to handle SCEVInit nodes and n-ary expressions.
  // For multiplication with init nodes, we create n-ary multiply expressions.

  if (const auto *lhsUnknown = dynamic_cast<const SCEVUnknown *>(lhsOperand),
      *rhsUnknown = dynamic_cast<const SCEVUnknown *>(rhsOperand);
      lhsUnknown || rhsUnknown)
  {
    return std::make_unique<SCEVUnknown>();
  }

  const auto lhsNAryAddExpr = dynamic_cast<const SCEVNAryAddExpr *>(lhsOperand);
  const auto rhsNAryAddExpr = dynamic_cast<const SCEVNAryAddExpr *>(rhsOperand);
  if (lhsNAryAddExpr || rhsNAryAddExpr)
  {
    // Handle n-ary add expressions - distribute multiplication
    // (a + b + c) × G = a×G + b×G + c×G
    const auto nAryAddExpr = lhsNAryAddExpr ? lhsNAryAddExpr : rhsNAryAddExpr;
    const auto other = lhsNAryAddExpr ? rhsOperand : lhsOperand;

    auto resultAddExpr = std::make_unique<SCEVNAryAddExpr>();
    for (auto operand : nAryAddExpr->GetOperands())
    {
      auto product = ApplyMulFolding(operand, other);
      resultAddExpr->AddOperand(std::move(product));
    }
    return resultAddExpr;
  }

  const auto lhsInit = dynamic_cast<const SCEVInit *>(lhsOperand);
  const auto rhsInit = dynamic_cast<const SCEVInit *>(rhsOperand);
  if (lhsInit && rhsInit)
  {
    // Two init nodes - create n-ary multiply expression
    return std::make_unique<SCEVNAryMulExpr>(lhsInit->Clone(), rhsInit->Clone());
  }

  const auto lhsNAryMulExpr = dynamic_cast<const SCEVNAryMulExpr *>(lhsOperand);
  const auto rhsNAryMulExpr = dynamic_cast<const SCEVNAryMulExpr *>(rhsOperand);
  if ((lhsInit && rhsNAryMulExpr) || (rhsInit && lhsNAryMulExpr))
  {
    // Init node with n-ary multiply expression - Clone mult expr and add init as an operand
    const auto * init = lhsInit ? lhsInit : rhsInit;
    auto * nAryMulExpr = lhsNAryMulExpr ? lhsNAryMulExpr : rhsNAryMulExpr;
    auto nAryMulExprClone = nAryMulExpr->Clone();
    auto newNAryMulExpr = dynamic_cast<SCEVNAryMulExpr *>(nAryMulExprClone.get());
    newNAryMulExpr->AddOperand(init->Clone());
    return newNAryMulExpr->Clone();
  }

  const auto lhsConstant = dynamic_cast<const SCEVConstant *>(lhsOperand);
  const auto rhsConstant = dynamic_cast<const SCEVConstant *>(rhsOperand);
  if ((lhsInit && rhsConstant && rhsConstant->GetValue() != 1)
      || (rhsInit && lhsConstant && lhsConstant->GetValue() != 1))
  {
    // Init node with non-one constant - create n-ary multiply expression
    const auto * init = lhsInit ? lhsInit : rhsInit;
    const auto * constant = lhsConstant ? lhsConstant : rhsConstant;
    return std::make_unique<SCEVNAryMulExpr>(init->Clone(), constant->Clone());
  }

  if (lhsInit || rhsInit)
  {
    // Single init node, no folding necessary
    const auto * init = lhsInit ? lhsInit : rhsInit;
    return init->Clone();
  }

  if (lhsNAryMulExpr && rhsNAryMulExpr)
  {
    // Two n-ary mult expressions - combine operands
    auto lhsNAryMulExprClone = lhsNAryMulExpr->Clone();
    auto lhsNewNAryMulExpr = dynamic_cast<SCEVNAryMulExpr *>(lhsNAryMulExprClone.get());
    for (auto op : rhsNAryMulExpr->GetOperands())
    {
      lhsNewNAryMulExpr->AddOperand(op->Clone());
    }
    return lhsNewNAryMulExpr->Clone();
  }

  if ((lhsNAryMulExpr && rhsConstant && rhsConstant->GetValue() != 1)
      || (rhsNAryMulExpr && lhsConstant && lhsConstant->GetValue() != 1))
  {
    // N-ary mult expression with non-one constant - Clone mult expression and add constant
    auto * nAryMulExpr = lhsNAryMulExpr ? lhsNAryMulExpr : rhsNAryMulExpr;
    auto * constant = lhsConstant ? lhsConstant : rhsConstant;

    auto nAryMulExprClone = nAryMulExpr->Clone();
    auto newNAryMulExpr = dynamic_cast<SCEVNAryMulExpr *>(nAryMulExprClone.get());
    newNAryMulExpr->AddOperand(constant->Clone());
    return newNAryMulExpr->Clone();
  }

  if (lhsNAryMulExpr || rhsNAryMulExpr)
  {
    const auto * nAryMulExpr = lhsNAryMulExpr ? lhsNAryMulExpr : rhsNAryMulExpr;
    return nAryMulExpr->Clone();
  }

  if (lhsConstant && rhsConstant)
  {
    // Two constants - fold by multiplying values together
    const auto lhsValue = lhsConstant->GetValue();
    const auto rhsValue = rhsConstant->GetValue();
    return std::make_unique<SCEVConstant>(lhsValue * rhsValue);
  }

  if (lhsConstant || rhsConstant)
  {
    const auto * constant = lhsConstant ? lhsConstant : rhsConstant;
    return constant->Clone();
  }

  return std::make_unique<SCEVUnknown>();
}

bool
ScalarEvolution::IsValidInductionVariable(
    const rvsdg::Output & variable,
    IVDependencyGraph & dependencyGraph)
{
  // First check that variable has only one self-reference,
  if (dependencyGraph[&variable][&variable].count != 1)
    return false;

  // Check that it has no reference via a mult-operation
  // (results in a geometric sequence - which we treat as an invalid induction variable)
  auto deps = dependencyGraph[&variable];
  for (auto [output, dependencyInfo] : deps)
  {
    if (dependencyInfo.operation == DependencyOp::Mul)
    {
      return false;
    }
  }

  // Then check for cycles through other variables
  std::unordered_set<const rvsdg::Output *> visited{};
  std::unordered_set<const rvsdg::Output *> recursionStack{};
  return !HasCycleThroughOthers(variable, variable, dependencyGraph, visited, recursionStack);
}

bool
ScalarEvolution::HasCycleThroughOthers(
    const rvsdg::Output & currentIV,
    const rvsdg::Output & originalIV,
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

    // Found a cycle back to the ORIGINAL node we started from
    // This means the original IV is explicitly part of the cycle
    if (depPtr == &originalIV)
      return true;

    // Already explored this branch, no cycle containing the original IV
    if (visited.find(depPtr) != visited.end())
      continue;

    // Recursively check dependencies, keeping track of the original node
    if (HasCycleThroughOthers(*depPtr, originalIV, dependencyGraph, visited, recursionStack))
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

  if (dynamic_cast<const SCEVUnknown *>(&a))
    return true;

  if (auto * constantA = dynamic_cast<const SCEVConstant *>(&a))
  {
    auto * constantB = dynamic_cast<const SCEVConstant *>(&b);
    return constantA->GetValue() == constantB->GetValue();
  }

  if (auto * initA = dynamic_cast<const SCEVInit *>(&a))
  {
    auto * initB = dynamic_cast<const SCEVInit *>(&b);
    return initA->GetPrePointer() == initB->GetPrePointer();
  }

  if (auto * binaryExprA = dynamic_cast<const SCEVBinaryExpr *>(&a))
  {
    auto * binaryExprB = dynamic_cast<const SCEVBinaryExpr *>(&b);
    return StructurallyEqual(*binaryExprA->GetLeftOperand(), *binaryExprB->GetLeftOperand())
        && StructurallyEqual(*binaryExprA->GetRightOperand(), *binaryExprB->GetRightOperand());
  }

  if (auto * chrecA = dynamic_cast<const SCEVChainRecurrence *>(&a))
  {
    auto * chrecB = dynamic_cast<const SCEVChainRecurrence *>(&b);
    if (chrecA->GetLoop() != chrecB->GetLoop())
      return false;
    if (chrecA->GetOperands().size() != chrecB->GetOperands().size())
      return false;
    for (size_t i = 0; i < chrecA->GetOperands().size(); ++i)
    {
      if (!StructurallyEqual(*chrecA->GetOperands()[i], *chrecB->GetOperands()[i]))
        return false;
    }
    return true;
  }

  if (auto * nAryExprA = dynamic_cast<const SCEVNAryExpr *>(&a))
  {
    auto * nAryExprB = dynamic_cast<const SCEVNAryExpr *>(&b);
    if (nAryExprA->GetOperands().size() != nAryExprB->GetOperands().size())
      return false;
    for (size_t i = 0; i < nAryExprA->GetOperands().size(); ++i)
    {
      if (!StructurallyEqual(*nAryExprA->GetOperands()[i], *nAryExprB->GetOperands()[i]))
        return false;
    }
    return true;
  }

  return false;
}
}

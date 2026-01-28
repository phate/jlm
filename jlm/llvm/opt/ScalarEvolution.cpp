/*
 * Copyright 2025 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/Trace.hpp>
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
  JLM_ASSERT(ptr);
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
    if (static_cast<int>(chrec->GetOperands().size()) == n + 1 && !IsUnknown(*chrec))
      count++;
  }
  return count;
}

size_t
ScalarEvolution::Context::GetNumTotalChrecs() const
{
  int count = 0;
  for (auto & [out, chrec] : ChrecMap_)
  {
    // Only count chrecs that are not unknown
    if (!IsUnknown(*chrec))
      count++;
  }
  return count;
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
  CombineChrecsAcrossLoops(*ctx);

  statistics->Stop(*ctx);
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
};

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

void
ScalarEvolution::CombineChrecsAcrossLoops(Context & ctx)
{
  bool changed{};
  do
  {
    changed = false;
    for (const auto & [output, chrec] : ctx.GetChrecs())
    {
      if (auto newSCEV = TryReplaceInitForSCEV(*chrec, ctx))
      {
        // Check if the result is actually a chrec
        if (dynamic_cast<const SCEVChainRecurrence *>(newSCEV->get()))
        {
          ctx.InsertChrec(*output, clone_as<SCEVChainRecurrence>(**newSCEV));
        }
        else
        {
          // The transformation produced a non-chrec SCEV (n-ary expression), store it in the SCEV
          // map instead
          ctx.InsertSCEV(*output, std::move(*newSCEV));
        }
        changed = true;
      }
    }
  } while (changed);
}

std::optional<std::unique_ptr<SCEV>>
ScalarEvolution::TryReplaceInitForSCEV(const SCEV & scev, Context & ctx)
{
  // This method is used to try to recursively find Init nodes in finalized recurrenes and their
  // corresponding chain recurrences (computed from other loops). It replaces the Init nodes with
  // their corresponding recurrence, and returns the resulting recurrence. In the case where no
  // change is made it returns nothing (nullopt)
  if (const auto initSCEV = dynamic_cast<const SCEVInit *>(&scev))
  {
    // Found an Init node, find the origin of its input value and get or create its chain recurrence
    const auto initPrePointer = initSCEV->GetPrePointer();
    if (const auto innerTheta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*initPrePointer))
    {
      const auto correspondingInput = innerTheta->MapPreLoopVar(*initPrePointer).input;
      const auto & inputOrigin = llvm::traceOutput(*correspondingInput->origin());
      if (const auto originSCEV = ctx.TryGetSCEVForOutput(inputOrigin))
      {
        // We have found a SCEV for the origin of the input, find the corresponding theta node so we
        // can create a recurrence for it
        const auto thetaParent = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(inputOrigin);
        const auto outerTheta = thetaParent
                                  ? thetaParent
                                  : dynamic_cast<rvsdg::ThetaNode *>(inputOrigin.region()->node());

        JLM_ASSERT(outerTheta);

        const auto chrec = GetOrCreateChainRecurrence(inputOrigin, *originSCEV, *outerTheta, ctx);

        // Create a chain recurrence for the SCEV, with the outer theta as the loop
        return chrec->Clone();
      }
    }
  }
  if (const auto nArySCEV = dynamic_cast<const SCEVNAryExpr *>(&scev))
  {
    // An n-ary scev is any scev with an arbitrary number of operands: chain recurrence, n-ary add
    // and n-ary mult. We want to recursively check all it's operands for Init nodes
    auto clone = clone_as<SCEVNAryExpr>(*nArySCEV);
    const auto operands = nArySCEV->GetOperands();
    bool changed = false;
    for (size_t i = 0; i < operands.size(); ++i)
    {
      if (auto result = TryReplaceInitForSCEV(*operands[i], ctx))
      {
        if (*result)
        {
          // Replace the Init operand with the chrec
          changed = true;
          clone->SwapOperand(i, std::move(*result));
        }
      }
    }
    if (!changed)
      return std::nullopt;

    if (dynamic_cast<const SCEVChainRecurrence *>(&scev))
    {
      // Result is a new chain recurrence, return it
      return clone;
    }

    if (dynamic_cast<const SCEVNAryExpr *>(&scev))
    {
      // If it is a n-ary expression, we try to fold the operands into themselves, e.g. if, after
      // replacing Init nodes with recurrences, we have ({0,+,1} + {1,+,2}) in an n-ary add
      // expression, we can fold this into {1,+,3}.
      bool folded{};
      do
      {
        folded = false;
        for (size_t i = 1; i < clone->GetOperands().size(); ++i)
        {
          std::vector<const SCEV *> ops = clone->GetOperands();
          const auto firstOp = ops[i - 1];
          const auto secondOp = ops[i];
          std::unique_ptr<SCEV> foldedOperand{};
          if (dynamic_cast<const SCEVNAryAddExpr *>(&scev))
          {
            foldedOperand = ApplyAddFolding(firstOp, secondOp);
          }
          else
          {
            foldedOperand = ApplyMulFolding(firstOp, secondOp);
          }

          if (foldedOperand)
          {
            clone->RemoveOperand(i);
            clone->SwapOperand(i - 1, foldedOperand);

            folded = true;
          }
        }
      } while (folded);

      if (clone->GetOperands().size() == 1)
      {
        // If there is only one operand in the n-ary expression, we just return the operand
        return clone->GetOperand(0)->Clone();
      }
    }

    return clone;
  }
  // Default is to just return nothing
  return std::nullopt;
}

void
ScalarEvolution::PerformSCEVAnalysis(const rvsdg::ThetaNode & thetaNode, Context & ctx)
{
  std::vector<rvsdg::ThetaNode::LoopVar> nonStateLoopVars;
  for (const auto loopVar : thetaNode.GetLoopVars())
  {
    if (loopVar.pre->Type()->Kind() != rvsdg::TypeKind::State)
    {
      nonStateLoopVars.push_back(loopVar);
    }
  }

  for (const auto loopVar : nonStateLoopVars)
  {
    const auto post = loopVar.post;
    auto scev = GetOrCreateSCEVForOutput(*post->origin(), ctx);
    ctx.InsertSCEV(*loopVar.output, scev); // Save the SCEV at the theta outputs as well
  }
  auto dependencyGraph = CreateDependencyGraph(nonStateLoopVars, ctx);

  util::HashSet<const rvsdg::Output *> validIVs{};
  for (const auto loopVar : nonStateLoopVars)
  {
    if (dependencyGraph[loopVar.pre].size() == 0)
    {
      // If the expression doesn't depend on at least one loop variable (including itself), it is
      // not an induction variable. Replace it with a SCEVUnknown
      ctx.InsertSCEV(*loopVar.post->origin(), SCEVUnknown::Create());
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
  for (const auto loopVar : nonStateLoopVars)
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
    const auto loopVarPost = loopVar.post;
    const auto scev = ctx.TryGetSCEVForOutput(*loopVarPost->origin());

    JLM_ASSERT(scev);
    ctx.InsertChrec(*loopVarPre, GetOrCreateChainRecurrence(*loopVarPre, *scev, thetaNode, ctx));
  }

  for (size_t i = order.size(); i < allVars.size(); ++i)
  {
    // Handle invalid induction variables
    const auto loopVarPre = allVars[i];
    auto unknownChainRecurrence = SCEVChainRecurrence::Create(thetaNode);
    unknownChainRecurrence->AddOperand(SCEVUnknown::Create());
    ctx.InsertChrec(*loopVarPre, unknownChainRecurrence);
  }
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
    result = SCEVPlaceholder::Create(output);
  }
  if (const auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
  {
    if (rvsdg::is<IntegerConstantOperation>(simpleNode->GetOperation()))
    {
      const auto constOp =
          dynamic_cast<const IntegerConstantOperation *>(&simpleNode->GetOperation());
      const auto value = constOp->Representation().to_int();
      result = SCEVConstant::Create(value);
    }
    if (rvsdg::is<IntegerBinaryOperation>(simpleNode->GetOperation()))
    {
      JLM_ASSERT(simpleNode->ninputs() == 2);
      const auto lhs = simpleNode->input(0)->origin();
      const auto rhs = simpleNode->input(1)->origin();

      auto lhsScev = GetOrCreateSCEVForOutput(*lhs, ctx);
      auto rhsScev = GetOrCreateSCEVForOutput(*rhs, ctx);
      if (rvsdg::is<IntegerAddOperation>(simpleNode->GetOperation()))
      {
        result = SCEVAddExpr::Create(std::move(lhsScev), std::move(rhsScev));
      }
      if (rvsdg::is<IntegerSubOperation>(simpleNode->GetOperation()))
      {
        auto rhsNegativeScev = GetNegativeSCEV(*rhsScev);

        result = SCEVAddExpr::Create(std::move(lhsScev), std::move(rhsNegativeScev));
      }
      if (rvsdg::is<IntegerMulOperation>(simpleNode->GetOperation()))
      {
        result = SCEVMulExpr::Create(std::move(lhsScev), std::move(rhsScev));
      }
    }
  }

  if (!result)
    // If none of the cases match, return an unknown SCEV expression
    result = SCEVUnknown::Create();

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
ScalarEvolution::CreateDependencyGraph(
    const std::vector<rvsdg::ThetaNode::LoopVar> & loopVars,
    const Context & ctx)
{
  IVDependencyGraph graph{};
  for (const auto loopVar : loopVars)
  {
    const auto post = loopVar.post;
    if (const auto scev = ctx.TryGetSCEVForOutput(*post->origin()))
    {
      DependencyMap dependencies{};

      FindDependenciesForSCEV(*scev.get(), dependencies);

      const auto pre = loopVar.pre;
      graph[pre] = dependencies;
    }
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
  JLM_ASSERT(result.size() == numVertices);
  return result;
}

std::unique_ptr<SCEVChainRecurrence>
ScalarEvolution::GetOrCreateChainRecurrence(
    const rvsdg::Output & output,
    const SCEV & scev,
    const rvsdg::ThetaNode & thetaNode,
    Context & ctx)
{
  if (const auto existing = ctx.TryGetChrecForOutput(output))
  {
    return clone_as<SCEVChainRecurrence>(*existing);
  }

  auto stepRecurrence = GetOrCreateStepForSCEV(output, scev, thetaNode, ctx);

  if (rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    // Find the start value for the recurrence
    const auto inputOrigin = thetaNode.MapPreLoopVar(output).input->origin();
    if (const auto constantInteger = tryGetConstantSignedInteger(*inputOrigin))
    {
      // If the input value is a constant, create a SCEV representation and set it as start
      // value (first operand in rec)
      stepRecurrence->AddOperandToFront(SCEVConstant::Create(*constantInteger));
    }
    else
    {
      // If not, create a SCEVInit node representing the start value
      stepRecurrence->AddOperandToFront(SCEVInit::Create(output));
    }
  }
  return stepRecurrence;
}

std::unique_ptr<SCEVChainRecurrence>
ScalarEvolution::GetOrCreateStepForSCEV(
    const rvsdg::Output & output,
    const SCEV & scevTree,
    const rvsdg::ThetaNode & thetaNode,
    Context & ctx)
{
  if (const auto existing = ctx.TryGetChrecForOutput(output))
  {
    return clone_as<SCEVChainRecurrence>(*existing);
  }

  auto chrec = SCEVChainRecurrence::Create(thetaNode);

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
    chrec->AddOperand(SCEVUnknown::Create());
    return chrec;
  }
  if (const auto scevAddExpr = dynamic_cast<const SCEVAddExpr *>(&scevTree))
  {
    const auto lhsStep =
        GetOrCreateStepForSCEV(output, *scevAddExpr->GetLeftOperand(), thetaNode, ctx);
    const auto rhsStep =
        GetOrCreateStepForSCEV(output, *scevAddExpr->GetRightOperand(), thetaNode, ctx);

    return clone_as<SCEVChainRecurrence>(*ApplyAddFolding(lhsStep.get(), rhsStep.get()));
  }
  if (const auto scevMulExpr = dynamic_cast<const SCEVMulExpr *>(&scevTree))
  {
    const auto lhsStep =
        GetOrCreateStepForSCEV(output, *scevMulExpr->GetLeftOperand(), thetaNode, ctx);
    const auto rhsStep =
        GetOrCreateStepForSCEV(output, *scevMulExpr->GetRightOperand(), thetaNode, ctx);

    return clone_as<SCEVChainRecurrence>(*ApplyMulFolding(lhsStep.get(), rhsStep.get()));
  }
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
    return SCEVUnknown::Create();
  }

  const auto lhsChrec = dynamic_cast<const SCEVChainRecurrence *>(lhsOperand);
  const auto rhsChrec = dynamic_cast<const SCEVChainRecurrence *>(rhsOperand);
  if (lhsChrec && rhsChrec)
  {
    if (lhsChrec->GetLoop() != rhsChrec->GetLoop())
    {
      return SCEVNAryAddExpr::Create(lhsChrec->Clone(), rhsChrec->Clone());
    }

    auto newChrec = SCEVChainRecurrence::Create(*lhsChrec->GetLoop());
    const auto lhsSize = lhsChrec->GetOperands().size();
    const auto rhsSize = rhsChrec->GetOperands().size();
    for (size_t i = 0; i < std::max(lhsSize, rhsSize); ++i)
    {
      const SCEV * lhs{};
      const SCEV * rhs{};
      if (i < lhsSize)
        lhs = lhsChrec->GetOperand(i);

      if (i < rhsSize)
        rhs = rhsChrec->GetOperand(i);
      newChrec->AddOperand(ApplyAddFolding(lhs, rhs));
    }
    return newChrec;
  }

  // Chrec + any other operand
  // This handles Init, Constant, and any other SCEV type uniformly
  if (lhsChrec || rhsChrec)
  {
    auto * chrec = lhsChrec ? lhsChrec : rhsChrec;
    const auto * otherOperand = lhsChrec ? rhsOperand : lhsOperand;

    // Skip if otherOperand is zero constant (identity for addition)
    if (const auto constant = dynamic_cast<const SCEVConstant *>(otherOperand))
    {
      if (!isNonZeroConstant(constant))
      {
        return chrec->Clone();
      }
    }
    auto newChrec = SCEVChainRecurrence::Create(*chrec->GetLoop());
    const auto chrecOperands = chrec->GetOperands();

    for (size_t i = 0; i < chrecOperands.size(); ++i)
    {
      const auto operand = chrecOperands[i];
      if (i == 0)
      {
        // Recursively fold the start value with the other operand
        newChrec->AddOperand(ApplyAddFolding(operand, otherOperand));
      }
      else
      {
        newChrec->AddOperand(operand->Clone());
      }
    }
    return newChrec;
  }

  const auto lhsNAryMulExpr = dynamic_cast<const SCEVNAryMulExpr *>(lhsOperand);
  const auto rhsNAryMulExpr = dynamic_cast<const SCEVNAryMulExpr *>(rhsOperand);
  // Handle n-ary multiply expressions - they become terms in an n-ary add expression
  if (lhsNAryMulExpr && rhsNAryMulExpr)
  {
    // Two multiply expressions - create add expression with both
    return SCEVNAryAddExpr::Create(lhsNAryMulExpr->Clone(), rhsNAryMulExpr->Clone());
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
    return SCEVNAryAddExpr::Create(mulExpr->Clone(), init->Clone());
  }

  const auto lhsConstant = dynamic_cast<const SCEVConstant *>(lhsOperand);
  const auto rhsConstant = dynamic_cast<const SCEVConstant *>(rhsOperand);
  if ((lhsNAryMulExpr && isNonZeroConstant(rhsConstant))
      || (rhsNAryMulExpr && isNonZeroConstant(lhsConstant)))
  {
    // Multiply expression with nonzero constant - create add expression
    const auto * mulExpr = lhsNAryMulExpr ? lhsNAryMulExpr : rhsNAryMulExpr;
    const auto * constant = lhsConstant ? lhsConstant : rhsConstant;
    return SCEVNAryAddExpr::Create(mulExpr->Clone(), constant->Clone());
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
    return SCEVNAryAddExpr::Create(lhsInit->Clone(), rhsInit->Clone());
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
    return SCEVNAryAddExpr::Create(init->Clone(), constant->Clone());
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

    return SCEVConstant::Create(lhsValue + rhsValue);
  }

  if (lhsConstant || rhsConstant)
  {
    const auto * constant = lhsConstant ? lhsConstant : rhsConstant;
    return constant->Clone();
  }

  return SCEVUnknown::Create();
}

std::unique_ptr<SCEV>
ScalarEvolution::ApplyMulFolding(const SCEV * lhsOperand, const SCEV * rhsOperand)
{
  // Apply folding rules for multiplication
  //
  // We have the following folding rules from the CR algebra:
  // G * {e,+,f}         =>       {G * e,+,G * f}
  // {e,+,f} * {g,+,h}   =>       {e * g,+,e * h + f * g + f * h,+,2*f*h}
  //
  // Similar to addition, we need to handle SCEVInit nodes and n-ary expressions.
  // For multiplication with init nodes, we create n-ary multiply expressions.

  if (const auto *lhsUnknown = dynamic_cast<const SCEVUnknown *>(lhsOperand),
      *rhsUnknown = dynamic_cast<const SCEVUnknown *>(rhsOperand);
      lhsUnknown || rhsUnknown)
  {
    return SCEVUnknown::Create();
  }

  const auto lhsChrec = dynamic_cast<const SCEVChainRecurrence *>(lhsOperand);
  const auto rhsChrec = dynamic_cast<const SCEVChainRecurrence *>(rhsOperand);
  if (lhsChrec && rhsChrec)
  {
    if (lhsChrec->GetLoop() != rhsChrec->GetLoop())
    {
      return SCEVNAryMulExpr::Create(lhsChrec->Clone(), rhsChrec->Clone());
    }

    auto newChrec = SCEVChainRecurrence::Create(*lhsChrec->GetLoop());
    const auto lhsSize = lhsChrec->GetOperands().size();
    const auto rhsSize = rhsChrec->GetOperands().size();

    if (lhsSize == 0)
    {
      for (auto operand : rhsChrec->GetOperands())
      {
        newChrec->AddOperand(operand->Clone());
      }
    }
    else if (rhsSize == 0)
    {
      for (auto operand : lhsChrec->GetOperands())
      {
        newChrec->AddOperand(operand->Clone());
      }
    }
    // Handle G * {e,+,f,...} where G is loop invariant
    if (lhsSize == 1)
    {
      // G * {e,+,f,...} = {G * e,+,G * f,...}
      auto lhs = lhsChrec->GetOperand(0);

      for (auto rhs : rhsChrec->GetOperands())
      {
        newChrec->AddOperand(ApplyMulFolding(lhs, rhs));
      }
    }
    else if (rhsSize == 1)
    {
      // {e,+,f,...} * G = {e * G,+,f * G,...}
      auto rhs = rhsChrec->GetOperand(0);

      for (auto lhs : lhsChrec->GetOperands())
      {
        newChrec->AddOperand(ApplyMulFolding(lhs, rhs));
      }
    }
    else if (lhsSize == 2 && rhsSize == 2)
    {
      // {e,+,f} * {g,+,h} = {e*g,+,e*h + f*g + f*h,+,2*f*h}
      const auto e = lhsChrec->GetOperand(0);
      const auto f = lhsChrec->GetOperand(1);
      const auto g = rhsChrec->GetOperand(0);
      const auto h = rhsChrec->GetOperand(1);

      // First step: e * g
      newChrec->AddOperand(ApplyMulFolding(e, g));

      // Second step: e * h + f * g + f * h
      const auto eh = ApplyMulFolding(e, h);
      const auto fg = ApplyMulFolding(f, g);
      const auto fh = ApplyMulFolding(f, h);
      const auto sum1 = ApplyAddFolding(eh.get(), fg.get());
      auto sum2 = ApplyAddFolding(sum1.get(), fh.get());
      newChrec->AddOperand(std::move(sum2));

      // Third step: 2 * f * h
      const auto two = SCEVConstant::Create(2);
      newChrec->AddOperand(ApplyMulFolding(two.get(), fh.get()));
    }
    else
    {
      // For other cases, return unknown
      newChrec->AddOperand(SCEVUnknown::Create());
    }
    return newChrec;
  }

  // Chrec * any other operand
  // This handles Init, Constant, and any other SCEV type uniformly
  if (lhsChrec || rhsChrec)
  {
    auto * chrec = lhsChrec ? lhsChrec : rhsChrec;
    const auto * otherOperand = lhsChrec ? rhsOperand : lhsOperand;

    // Skip if other operand is constant one (identity for multiplication)
    if (auto constant = dynamic_cast<const SCEVConstant *>(otherOperand))
    {
      if (constant->GetValue() == 1)
      {
        return chrec->Clone();
      }
    }
    auto newChrec = SCEVChainRecurrence::Create(*chrec->GetLoop());
    auto chrecOperands = chrec->GetOperands();

    for (size_t i = 0; i < chrecOperands.size(); ++i)
    {
      auto operand = chrecOperands[i];
      // Recursively fold the start value with the other operand
      newChrec->AddOperand(ApplyMulFolding(operand, otherOperand));
    }
    return newChrec;
  }

  const auto lhsNAryAddExpr = dynamic_cast<const SCEVNAryAddExpr *>(lhsOperand);
  const auto rhsNAryAddExpr = dynamic_cast<const SCEVNAryAddExpr *>(rhsOperand);
  if (lhsNAryAddExpr || rhsNAryAddExpr)
  {
    // Handle n-ary add expressions - distribute multiplication
    // (a + b + c) × G = a×G + b×G + c×G
    const auto nAryAddExpr = lhsNAryAddExpr ? lhsNAryAddExpr : rhsNAryAddExpr;
    const auto other = lhsNAryAddExpr ? rhsOperand : lhsOperand;

    auto resultAddExpr = SCEVNAryAddExpr::Create();
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
    return SCEVNAryMulExpr::Create(lhsInit->Clone(), rhsInit->Clone());
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
    return SCEVNAryMulExpr::Create(init->Clone(), constant->Clone());
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
    return SCEVConstant::Create(lhsValue * rhsValue);
  }

  if (lhsConstant || rhsConstant)
  {
    const auto * constant = lhsConstant ? lhsConstant : rhsConstant;
    return constant->Clone();
  }

  return SCEVUnknown::Create();
}

std::unique_ptr<SCEV>
ScalarEvolution::GetNegativeSCEV(const SCEV & scev)
{
  // -(c)
  if (const auto c = dynamic_cast<const SCEVConstant *>(&scev))
  {
    const auto value = c->GetValue();
    return SCEVConstant::Create(-value);
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
    return SCEVAddExpr::Create(
        GetNegativeSCEV(*add->GetLeftOperand()),
        GetNegativeSCEV(*add->GetRightOperand()));
  }
  // General case: -(x) -> (-1) * x
  return SCEVMulExpr::Create(SCEVConstant::Create(-1), scev.Clone());
}

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

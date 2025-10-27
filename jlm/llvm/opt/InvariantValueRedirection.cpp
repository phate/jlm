/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/FunctionPointer.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/InvariantValueRedirection.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{

class InvariantValueRedirection::Statistics final : public util::Statistics
{
public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Statistics::Id::InvariantValueRedirection, sourceFile)
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

InvariantValueRedirection::~InvariantValueRedirection() = default;

void
InvariantValueRedirection::Run(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(module.SourceFilePath().value());

  statistics->Start();
  RedirectInRootRegion(module.Rvsdg());
  statistics->Stop();

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

void
InvariantValueRedirection::RedirectInRootRegion(rvsdg::Graph & rvsdg)
{
  // We require a topdown traversal in the root region to ensure that a lambda node is visited
  // before its call nodes. This ensures that all invariant values are redirected in the lambda
  // subregion before we try to detect invariant call outputs.
  for (auto node : rvsdg::TopDownTraverser(&rvsdg.GetRootRegion()))
  {
    MatchTypeOrFail(
        *node,
        [](const rvsdg::LambdaNode & lambdaNode)
        {
          RedirectInRegion(*lambdaNode.subregion());
        },
        [](const rvsdg::PhiNode & phiNode)
        {
          auto phiLambdaNodes = rvsdg::PhiNode::ExtractLambdaNodes(phiNode);
          for (auto phiLambdaNode : phiLambdaNodes)
          {
            RedirectInRegion(*phiLambdaNode->subregion());
          }
        },
        [](const rvsdg::DeltaNode &)
        {
          // Nothing needs to be done.
          // Delta nodes are irrelevant for invariant value redirection.
        },
        [](const rvsdg::SimpleNode & simpleNode)
        {
          MatchTypeOrFail(
              simpleNode.GetOperation(),
              [](const FunctionToPointerOperation &)
              {
                // Nothing needs to be done.
              },
              [](const PointerToFunctionOperation &)
              {
                // Nothing needs to be done.
              });
        });
  }
}

void
InvariantValueRedirection::RedirectInRegion(rvsdg::Region & region)
{
  const auto isGammaNode = !!dynamic_cast<rvsdg::GammaNode *>(region.node());
  const auto isThetaNode = !!dynamic_cast<rvsdg::ThetaNode *>(region.node());
  const auto isLambdaNode = !!dynamic_cast<rvsdg::LambdaNode *>(region.node());
  JLM_ASSERT(isGammaNode || isThetaNode || isLambdaNode);

  // We do not need a traverser here and can just iterate through all the nodes of a region as
  // it is irrelevant in which order we handle the nodes.
  for (auto & node : region.Nodes())
  {
    if (auto gammaNode = dynamic_cast<rvsdg::GammaNode *>(&node))
    {
      // Ensure we redirect invariant values of all nodes in the gamma subregions first, otherwise
      // we might not be able to redirect some of the gamma outputs.
      RedirectInSubregions(*gammaNode);
      RedirectGammaOutputs(*gammaNode);
    }
    else if (auto thetaNode = dynamic_cast<rvsdg::ThetaNode *>(&node))
    {
      // Ensure we redirect invariant values of all nodes in the theta subregion first, otherwise we
      // might not be able to redirect some of the theta outputs.
      RedirectInSubregions(*thetaNode);
      RedirectThetaOutputs(*thetaNode);
    }
    else if (auto simpleNode = dynamic_cast<rvsdg::SimpleNode *>(&node))
    {
      if (is<CallOperation>(simpleNode))
      {
        RedirectCallOutputs(*util::assertedCast<rvsdg::SimpleNode>(&node));
      }
    }
  }
}

void
InvariantValueRedirection::RedirectInSubregions(rvsdg::StructuralNode & structuralNode)
{
  const auto isGammaNode = !!dynamic_cast<rvsdg::GammaNode *>(&structuralNode);
  const auto isThetaNode = !!dynamic_cast<rvsdg::ThetaNode *>(&structuralNode);
  JLM_ASSERT(isGammaNode || isThetaNode);

  for (auto & subregion : structuralNode.Subregions())
  {
    RedirectInRegion(subregion);
  }
}

void
InvariantValueRedirection::RedirectGammaOutputs(rvsdg::GammaNode & gammaNode)
{
  for (auto exitvar : gammaNode.GetExitVars())
  {
    if (auto invariantOrigin = rvsdg::GetGammaInvariantOrigin(gammaNode, exitvar))
    {
      exitvar.output->divert_users(*invariantOrigin);
    }
  }
}

void
InvariantValueRedirection::RedirectThetaOutputs(rvsdg::ThetaNode & thetaNode)
{
  redirectThetaGammaOutputs(thetaNode);

  for (const auto & loopVar : thetaNode.GetLoopVars())
  {
    // FIXME: In order to also redirect I/O state type variables, we need to know whether a loop
    // terminates.
    if (rvsdg::is<IOStateType>(loopVar.input->Type()))
      continue;

    if (rvsdg::ThetaLoopVarIsInvariant(loopVar))
      loopVar.output->divert_users(loopVar.input->origin());
  }
}

void
InvariantValueRedirection::redirectThetaGammaOutputs(rvsdg::ThetaNode & thetaNode)
{
  const auto & thetaPredicateOperand = *thetaNode.predicate()->origin();
  const auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(thetaPredicateOperand);
  if (!gammaNode)
  {
    // The theta node predicate does not originate from a gamma node. Nothing can be done.
    return;
  }

  auto subregionRolesOpt = determineGammaSubregionRoles(*gammaNode, thetaPredicateOperand);
  if (!subregionRolesOpt.has_value())
  {
    // We could not determine the roles of the gamma subregions. Nothing can be done.
    return;
  }
  auto roles = *subregionRolesOpt;

  auto divertLoopVar =
      [&gammaNode](rvsdg::ThetaNode::LoopVar & loopVar, rvsdg::Output & entryVarArgument)
  {
    if (rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(entryVarArgument))
    {
      auto roleVar = gammaNode->MapBranchArgument(entryVarArgument);
      if (auto entryVar = std::get_if<rvsdg::GammaNode::EntryVar>(&roleVar))
      {
        loopVar.post->divert_to(entryVar->input->origin());
      }
    }
  };

  // At this point we can try to redirect the theta node loop variables
  for (auto & loopVar : thetaNode.GetLoopVars())
  {
    if (loopVar.output->IsDead() && loopVar.pre->IsDead())
    {
      // The loop variable is completely dead. We do not need to waste any effort on it.
      continue;
    }

    auto & loopVarPostOperand = *loopVar.post->origin();
    if (rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(loopVarPostOperand) != gammaNode)
    {
      // The post value of the loop variable does not originate from the gamma node. Nothing can
      // be done.
      continue;
    }
    auto [branchResult, _] = gammaNode->MapOutputExitVar(loopVarPostOperand);

    if (loopVar.output->IsDead())
    {
      // The loop variables' output is dead, which means only its repetition value is of interest.
      auto & entryVarArgument = *branchResult[roles.repetitionSubregion->index()]->origin();
      divertLoopVar(loopVar, entryVarArgument);
    }
    else if (loopVar.pre->IsDead())
    {
      // The loop variables' pre value is dead, which means only its exit value is of interest.
      auto & entryVarArgument = *branchResult[roles.exitSubregion->index()]->origin();
      divertLoopVar(loopVar, entryVarArgument);
    }
  }
}

std::optional<InvariantValueRedirection::GammaSubregionRoles>
InvariantValueRedirection::determineGammaSubregionRoles(
    rvsdg::GammaNode & gammaNode,
    const rvsdg::Output & thetaPredicateOperand)
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(thetaPredicateOperand) == &gammaNode);

  if (gammaNode.nsubregions() != 2)
  {
    return std::nullopt;
  }

  auto [branchResult, _] = gammaNode.MapOutputExitVar(thetaPredicateOperand);
  auto [constantNodeA, constantOperationA] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::ControlConstantOperation>(
          *branchResult[0]->origin());
  if (constantOperationA == nullptr)
  {
    return std::nullopt;
  }

  auto [constantNodeB, constantOperationB] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::ControlConstantOperation>(
          *branchResult[1]->origin());
  if (constantOperationB == nullptr)
  {
    return std::nullopt;
  }

  const size_t alternativeA = constantOperationA->value().alternative();
  const size_t alternativeB = constantOperationB->value().alternative();
  JLM_ASSERT(alternativeA == 0 || alternativeA == 1);
  JLM_ASSERT(alternativeB == 0 || alternativeB == 1);
  JLM_ASSERT(alternativeA != alternativeB);

  GammaSubregionRoles roles;
  if (alternativeA == 0)
  {
    roles.exitSubregion = constantNodeA->region();
    roles.repetitionSubregion = constantNodeB->region();
  }
  else
  {
    roles.exitSubregion = constantNodeB->region();
    roles.repetitionSubregion = constantNodeA->region();
  }

  return roles;
}

void
InvariantValueRedirection::RedirectCallOutputs(rvsdg::SimpleNode & callNode)
{
  JLM_ASSERT(is<CallOperation>(&callNode));

  auto callTypeClassifier = CallOperation::ClassifyCall(callNode);
  auto callType = callTypeClassifier->GetCallType();

  // FIXME: We currently only support non-recursive direct calls. We would also like to get this
  // working for recursive direct calls, but that requires a little bit more work as we need to be
  // able to break the cycles between the recursive calls.
  if (callType != CallTypeClassifier::CallType::NonRecursiveDirectCall)
    return;

  auto & lambdaNode =
      rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(callTypeClassifier->GetLambdaOutput());

  // LLVM permits code where it can happen that the number and type of arguments handed in to the
  // call node do not agree with the number and type of lambda parameters, even though it is a
  // direct call. See jlm::tests::LambdaCallArgumentMismatch for an example. In this case, we cannot
  // redirect the call outputs to the call operand as the types would not align, resulting in type
  // errors.
  if (CallOperation::NumArguments(callNode) != lambdaNode.GetFunctionArguments().size())
    return;

  const auto memoryStateOutput = &CallOperation::GetMemoryStateOutput(callNode);
  const auto callExitSplit = CallOperation::GetMemoryStateExitSplit(callNode);
  const auto callEntryMerge = CallOperation::GetMemoryStateEntryMerge(callNode);
  const auto lambdaEntrySplit = GetMemoryStateEntrySplit(lambdaNode);
  const auto lambdaExitMerge = GetMemoryStateExitMerge(lambdaNode);

  const auto hasAllMemoryStateNodes = callExitSplit != nullptr && callEntryMerge != nullptr
                                   && lambdaEntrySplit != nullptr && lambdaExitMerge != nullptr;

  const auto results = lambdaNode.GetFunctionResults();
  JLM_ASSERT(callNode.noutputs() == results.size());
  for (size_t n = 0; n < callNode.noutputs(); n++)
  {
    const auto callOutput = callNode.output(n);
    const auto shouldHandleMemoryStateOperations =
        (callOutput == memoryStateOutput) && hasAllMemoryStateNodes;

    if (shouldHandleMemoryStateOperations)
    {
      // All the inputs and outputs of the memory state nodes need to be aligned.
      JLM_ASSERT(callExitSplit->noutputs() == lambdaExitMerge->ninputs());
      JLM_ASSERT(lambdaExitMerge->ninputs() == lambdaEntrySplit->noutputs());
      JLM_ASSERT(lambdaEntrySplit->noutputs() == callEntryMerge->ninputs());

      for (auto & lambdaExitMergeInput : lambdaExitMerge->Inputs())
      {
        auto & lambdaEntrySplitOutput = *lambdaExitMergeInput.origin();
        const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(lambdaEntrySplitOutput);
        if (node != lambdaEntrySplit)
        {
          // The state edge is not invariant. Let's move on to the next one.
          continue;
        }

        const auto lambdaExitMemoryNodeId =
            LambdaExitMemoryStateMergeOperation::mapInputToMemoryNodeId(lambdaExitMergeInput);
        const auto callExitSplitOutput = CallExitMemoryStateSplitOperation::mapMemoryNodeIdToOutput(
            *callExitSplit,
            lambdaExitMemoryNodeId);

        const auto lambdaEntryMemoryNodeId =
            LambdaEntryMemoryStateSplitOperation::mapOutputToMemoryNodeId(lambdaEntrySplitOutput);
        const auto callEntryMergeInput = CallEntryMemoryStateMergeOperation::mapMemoryNodeIdToInput(
            *callEntryMerge,
            lambdaEntryMemoryNodeId);

        // We expect that the memory node IDs for a given state between a
        // LambdaEntryMemoryStateMergeOperation node and a LambdaExitMemoryStateSplitOperation node
        // are always the same, otherwise we have a bug in the memory state encoding.
        JLM_ASSERT(lambdaExitMemoryNodeId == lambdaEntryMemoryNodeId);

        if (callExitSplitOutput != nullptr && callEntryMergeInput != nullptr)
        {
          callExitSplitOutput->divert_users(callEntryMergeInput->origin());
        }
      }
    }
    else
    {
      auto & lambdaResult = *results[n];
      auto origin = lambdaResult.origin();
      if (rvsdg::TryGetRegionParentNode<rvsdg::LambdaNode>(*origin) == &lambdaNode)
      {
        if (auto ctxvar = lambdaNode.MapBinderContextVar(*origin))
        {
          // This is a bound context variable.
          // FIXME: We would like to get this case working as well, but we need to route the origin
          // of the respective lambda input to the subregion of the call node.
        }
        else
        {
          auto callOperand = CallOperation::Argument(callNode, origin->index())->origin();
          callOutput->divert_users(callOperand);
        }
      }
    }
  }
}

}

/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/FunctionPointer.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/InvariantValueRedirection.hpp>
#include <jlm/llvm/opt/PredicateCorrelation.hpp>
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
    rvsdg::MatchType(
        node,
        [](rvsdg::GammaNode & gammaNode)
        {
          // Ensure we redirect invariant values of all nodes in the gamma subregions first,
          // otherwise we might not be able to redirect some of the gamma outputs.
          RedirectInSubregions(gammaNode);
          RedirectGammaOutputs(gammaNode);
        },
        [](rvsdg::ThetaNode & thetaNode)
        {
          // Ensure we redirect invariant values of all nodes in the theta subregion first,
          // otherwise we might not be able to redirect some of the theta outputs.
          RedirectInSubregions(thetaNode);
          RedirectThetaOutputs(thetaNode);
        },
        [](rvsdg::SimpleNode & simpleNode)
        {
          rvsdg::MatchType(
              simpleNode.GetOperation(),
              [&simpleNode](const CallOperation &)
              {
                RedirectCallOutputs(simpleNode);
              },
              [&simpleNode](const LoadOperation &)
              {
                redirectLoadOutputs(simpleNode);
              });
        });
  }

  region.prune(false);
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
  for (auto exitVar : gammaNode.GetExitVars())
  {
    if (auto invariantOrigin = rvsdg::GetGammaInvariantOrigin(gammaNode, exitVar))
    {
      exitVar.output->divert_users(*invariantOrigin);
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
  auto correlationOpt = computeThetaGammaPredicateCorrelation(thetaNode);
  if (!correlationOpt.has_value())
  {
    return;
  }
  auto & correlation = correlationOpt.value();

  auto subregionRolesOpt = determineGammaSubregionRoles(*correlation);
  if (!subregionRolesOpt.has_value())
  {
    // We could not determine the roles of the gamma subregions. Nothing can be done.
    return;
  }
  auto roles = *subregionRolesOpt;
  auto & gammaNode = correlation->gammaNode();

  auto divertLoopVar =
      [&gammaNode](rvsdg::ThetaNode::LoopVar & loopVar, rvsdg::Output & entryVarArgument)
  {
    if (rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(entryVarArgument))
    {
      auto roleVar = gammaNode.MapBranchArgument(entryVarArgument);
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
    if (rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(loopVarPostOperand) != &gammaNode)
    {
      // The post value of the loop variable does not originate from the gamma node. Nothing can
      // be done.
      continue;
    }
    auto [branchResult, _] = gammaNode.MapOutputExitVar(loopVarPostOperand);

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

  // First, handle all call outputs where the corresponding function result is invariant
  const auto results = lambdaNode.GetFunctionResults();
  JLM_ASSERT(callNode.noutputs() == results.size());
  for (size_t n = 0; n < callNode.noutputs(); n++)
  {
    const auto callOutput = callNode.output(n);

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

  // Next, handle lambda bodies that contain memory state split and merge nodes.
  // Memory state edges can only be routed around the call if the corresponding memory node id
  // is invariant between the LambdaEntrySplit and the LambdaExitMerge.
  const auto callExitSplit = CallOperation::tryGetMemoryStateExitSplit(callNode);
  const auto callEntryMerge = CallOperation::tryGetMemoryStateEntryMerge(callNode);
  const auto lambdaEntrySplit = tryGetMemoryStateEntrySplit(lambdaNode);
  const auto lambdaExitMerge = tryGetMemoryStateExitMerge(lambdaNode);

  // Only continue if the call / lambda pair has all four memory state nodes
  if (callExitSplit == nullptr || callEntryMerge == nullptr || lambdaEntrySplit == nullptr
      || lambdaExitMerge == nullptr)
    return;

  const auto callExitSplitOp =
      *util::assertedCast<const CallExitMemoryStateSplitOperation>(&callExitSplit->GetOperation());
  for (const auto memoryNodeId : callExitSplitOp.getMemoryNodeIds())
  {
    const auto result = LambdaExitMemoryStateMergeOperation::tryMapMemoryNodeIdToInput(
        *lambdaExitMerge,
        memoryNodeId);
    const auto argument = LambdaEntryMemoryStateSplitOperation::tryMapMemoryNodeIdToOutput(
        *lambdaEntrySplit,
        memoryNodeId);

    // If the lambda does not route this memory state at all, it is effectively invariant
    if (result != nullptr && argument != nullptr)
    {
      // If the lambda body has an edge for this memory state, and it is not invariant, we must stop
      if (result->origin() != argument)
        continue;
    }

    // If we get here, the memory state is invariant, and can be routed around the call
    const auto output =
        CallExitMemoryStateSplitOperation::tryMapMemoryNodeIdToOutput(*callExitSplit, memoryNodeId);
    const auto input = CallEntryMemoryStateMergeOperation::tryMapMemoryNodeIdToInput(
        *callEntryMerge,
        memoryNodeId);

    JLM_ASSERT(output);
    if (!input)
      continue;

    output->divert_users(input->origin());
  }
}

void
InvariantValueRedirection::redirectLoadOutputs(rvsdg::SimpleNode & loadNode)
{
  if (LoadOperation::LoadedValueOutput(loadNode).IsDead())
  {
    for (auto & memoryStateOutput : LoadOperation::MemoryStateOutputs(loadNode))
    {
      auto & memoryStateInput = LoadOperation::MapMemoryStateOutputToInput(memoryStateOutput);
      memoryStateOutput.divert_users(memoryStateInput.origin());
    }
  }
}

}

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
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{

class InvariantValueRedirection::Statistics final : public util::Statistics
{
public:
  ~Statistics() override = default;

  explicit Statistics(const util::filepath & sourceFile)
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
  Create(const util::filepath & sourceFile)
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
    if (auto lambdaNode = dynamic_cast<rvsdg::LambdaNode *>(node))
    {
      RedirectInRegion(*lambdaNode->subregion());
    }
    else if (auto phiNode = dynamic_cast<phi::node *>(node))
    {
      auto phiLambdaNodes = phi::node::ExtractLambdaNodes(*phiNode);
      for (auto phiLambdaNode : phiLambdaNodes)
      {
        RedirectInRegion(*phiLambdaNode->subregion());
      }
    }
    else if (is<delta::operation>(node))
    {
      // Nothing needs to be done.
      // Delta nodes are irrelevant for invariant value redirection.
    }
    else if (
        is<FunctionToPointerOperation>(node->GetOperation())
        || is<PointerToFunctionOperation>(node->GetOperation()))
    {
      // Nothing needs to be done.
    }
    else
    {
      JLM_UNREACHABLE("Unhandled node type.");
    }
  }
}

void
InvariantValueRedirection::RedirectInRegion(rvsdg::Region & region)
{
  auto isGammaNode = is<rvsdg::GammaOperation>(region.node());
  auto isThetaNode = is<rvsdg::ThetaOperation>(region.node());
  auto isLambdaNode = is<rvsdg::LambdaOperation>(region.node());
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
    else if (is<CallOperation>(&node))
    {
      RedirectCallOutputs(*util::AssertedCast<rvsdg::SimpleNode>(&node));
    }
  }
}

void
InvariantValueRedirection::RedirectInSubregions(rvsdg::StructuralNode & structuralNode)
{
  auto isGammaNode = is<rvsdg::GammaOperation>(&structuralNode);
  auto isThetaNode = is<rvsdg::ThetaOperation>(&structuralNode);
  JLM_ASSERT(isGammaNode || isThetaNode);

  for (size_t n = 0; n < structuralNode.nsubregions(); n++)
  {
    RedirectInRegion(*structuralNode.subregion(n));
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
  for (const auto & loopVar : thetaNode.GetLoopVars())
  {
    // FIXME: In order to also redirect I/O state type variables, we need to know whether a loop
    // terminates.
    if (rvsdg::is<IOStateType>(loopVar.input->type()))
      continue;

    if (rvsdg::ThetaLoopVarIsInvariant(loopVar))
      loopVar.output->divert_users(loopVar.input->origin());
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

  auto memoryStateOutput = &CallOperation::GetMemoryStateOutput(callNode);
  auto callExitSplit = CallOperation::GetMemoryStateExitSplit(callNode);
  auto hasCallExitSplit = callExitSplit != nullptr;

  auto results = lambdaNode.GetFunctionResults();
  JLM_ASSERT(callNode.noutputs() == results.size());
  for (size_t n = 0; n < callNode.noutputs(); n++)
  {
    auto callOutput = callNode.output(n);
    auto shouldHandleMemoryStateOperations = (callOutput == memoryStateOutput) && hasCallExitSplit;

    if (shouldHandleMemoryStateOperations)
    {
      auto lambdaEntrySplit = GetMemoryStateEntrySplit(lambdaNode);
      auto lambdaExitMerge = GetMemoryStateExitMerge(lambdaNode);
      auto callEntryMerge = CallOperation::GetMemoryStateEntryMerge(callNode);

      // The callExitSplit is present. We therefore expect the other nodes to be present as well.
      JLM_ASSERT(lambdaEntrySplit && lambdaExitMerge && callEntryMerge);

      // All the inputs and outputs of the memory state nodes need to be aligned.
      JLM_ASSERT(callExitSplit->noutputs() == lambdaExitMerge->ninputs());
      JLM_ASSERT(lambdaExitMerge->ninputs() == lambdaEntrySplit->noutputs());
      JLM_ASSERT(lambdaEntrySplit->noutputs() == callEntryMerge->ninputs());

      for (size_t i = 0; i < lambdaExitMerge->ninputs(); i++)
      {
        auto lambdaExitMergeInput = lambdaExitMerge->input(i);
        auto node = rvsdg::output::GetNode(*lambdaExitMergeInput->origin());
        if (node == lambdaEntrySplit)
        {
          auto callExitSplitOutput = callExitSplit->output(lambdaExitMergeInput->index());
          auto callEntryMergeOperand =
              callEntryMerge->input(lambdaExitMergeInput->origin()->index())->origin();
          callExitSplitOutput->divert_users(callEntryMergeOperand);
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

/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/CallSummary.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/opt/inlining.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/Trace.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::llvm
{

class FunctionInlining::Statistics final : public util::Statistics
{
  // The total number of lambda nodes
  static constexpr const char * NumFunctions_ = "#Functions";
  // The total number of lambda nodes that are marked as possible to inline
  static constexpr const char * NumInlineableFunctions_ = "#InlineableFunctions";
  // The total number of call operations
  static constexpr const char * NumFunctionCalls_ = "#FunctionCalls";
  // The number of call operations that could in theory be inlined
  static constexpr const char * NumInlineableCalls_ = "#InlinableCalls";
  // The number of call operations that were actually inlined
  static constexpr const char * NumCallsInlined_ = "#CallsInlined";

public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Id::FunctionInlining, sourceFile)
  {}

  void
  start()
  {
    AddTimer(Label::Timer).start();
  }

  void
  stop(
      size_t numFunctions,
      size_t numInlineableFunctions,
      size_t numFunctionCalls,
      size_t numInlineableCalls,
      size_t numCallsInlined)
  {
    GetTimer(Label::Timer).stop();
    AddMeasurement(NumFunctions_, numFunctions);
    AddMeasurement(NumInlineableFunctions_, numInlineableFunctions);
    AddMeasurement(NumFunctionCalls_, numFunctionCalls);
    AddMeasurement(NumInlineableCalls_, numInlineableCalls);
    AddMeasurement(NumCallsInlined_, numCallsInlined);
  }

  static std::unique_ptr<Statistics>
  create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

struct FunctionInlining::Context
{
  // Functions that are possible to inline
  // Just because a function is on this list, does not mean it should be inlined
  util::HashSet<const rvsdg::LambdaNode *> inlineableFunctions;

  // Functions that are not exported from the module, and only called once
  util::HashSet<const rvsdg::LambdaNode *> functionsCalledOnce;

  // Used for statistics
  size_t numFunctions = 0;
  size_t numFunctionCalls = 0;
  size_t numInlineableCalls = 0;
  size_t numInlinedCalls = 0;
};

FunctionInlining::~FunctionInlining() noexcept = default;

FunctionInlining::FunctionInlining()
    : Transformation("FunctionInlining")
{}

/**
 * A function body has access to the function's context variables,
 * so inlining requires these context variables to be routed into the call's region.
 * @param region the region to make the context variables available in
 * @param callee the function being called
 * @return an output inside the given \p region, for each context variable in \p callee
 */
static std::vector<rvsdg::Output *>
routeContextVariablesToRegion(rvsdg::Region & region, const rvsdg::LambdaNode & callee)
{
  llvm::OutputTracer tracer;
  // We avoid entering phi nodes, as we can not route from a sibling region
  tracer.setEnterPhiNodes(false);

  std::vector<rvsdg::Output *> deps;
  for (auto & ctxvar : callee.GetContextVars())
  {
    auto & traced = tracer.trace(*ctxvar.input->origin());
    auto & routed = rvsdg::RouteToRegion(traced, region);
    deps.push_back(&routed);
  }

  return deps;
}

/**
 * After the body of a function has been copied to the region of a call node,
 * and inputs and outputs of the call have been routed to the copied body,
 * we may have a graph that looks like:
 *
 *       |       |       |
 *       V       V       V
 *      [4]     [7]     [9]
 *  CallEntryMemoryStateMerge
 *              |
 *              V
 *  LambdaEntryMemoryStateSplit
 *        [4]        [9]
 *         |          |
 *         V          V
 *
 *     [  rest of function  ]
 *     [   body goes here   ]
 *
 *        |          |
 *        V          V
 *       [4]        [9]
 *  LambdaExitMemoryStateMerge
 *             |
 *             V
 *   CallExitMemoryStateSplit
 *      [4]     [7]     [9]
 *       |       |       |
 *       V       V       V
 *
 * In these cases, we want to remove all the merge and split nodes, and route the relevant memory
 * states into the function body directly. Memory state IDs that are unused within the function,
 * such as [9] in the example above, are instead routed around from the call merge.
 * Memory states IDs that only exist inside the function body are given undef nodes.
 * @param callEntryMerge the call entry merge node in the diagram above
 * @param callExitSplit the call exit split node in the diagram above
 */
static void
tryRerouteMemoryStateMergeAndSplit(
    rvsdg::SimpleNode & callEntryMerge,
    rvsdg::SimpleNode & callExitSplit)
{
  const auto callEntryMergeOp =
      rvsdg::tryGetOperation<CallEntryMemoryStateMergeOperation>(callEntryMerge);
  const auto callExitSplitOp =
      rvsdg::tryGetOperation<CallExitMemoryStateSplitOperation>(callExitSplit);
  JLM_ASSERT(callEntryMergeOp);
  JLM_ASSERT(callExitSplitOp);

  // Use the output of the callEntryMerge to look for a lambdaEntrySplit
  auto & callEntryMergeOutput = *callEntryMerge.output(0);
  if (callEntryMergeOutput.nusers() != 1)
    return;
  auto & user = callEntryMergeOutput.SingleUser();
  const auto [lambdaEntrySplit, lambdaEntrySplitOp] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<LambdaEntryMemoryStateSplitOperation>(user);
  if (!lambdaEntrySplitOp)
    return;

  // Use the input of the callExitMerge to look for a lambdaExitMerge
  auto & callExitSplitInput = *callExitSplit.input(0)->origin();
  const auto [lambdaExitSplit, lambdaExitSplitOp] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<LambdaExitMemoryStateMergeOperation>(callExitSplitInput);
  if (!lambdaExitSplitOp)
    return;

  // For each memory state output of the lambdaEntrySplit, move its users or create undef nodes
  for (auto & output : lambdaEntrySplit->Outputs())
  {
    auto memoryStateId = LambdaEntryMemoryStateSplitOperation::mapOutputToMemoryNodeId(output);
    auto mergeInput = CallEntryMemoryStateMergeOperation::tryMapMemoryNodeIdToInput(
        callEntryMerge,
        memoryStateId);
    if (mergeInput)
    {
      output.divert_users(mergeInput->origin());
    }
    else
    {
      // The call has no matching memory state going into it, so we create an undef node
      const auto undef = UndefValueOperation::Create(*output.region(), output.Type());
      output.divert_users(undef);
    }
  }

  // For each memory state output of the callExitSplit, move its users
  for (auto & output : callExitSplit.Outputs())
  {
    auto memoryStateId = CallExitMemoryStateSplitOperation::mapOutputToMemoryNodeId(output);
    auto exitMergeInput = LambdaExitMemoryStateMergeOperation::tryMapMemoryNodeIdToInput(
        *lambdaExitSplit,
        memoryStateId);
    if (exitMergeInput)
    {
      output.divert_users(exitMergeInput->origin());
    }
    else
    {
      // the memory state id was never routed through the inside of the lambda, so route it around
      auto entryMergeInput = CallEntryMemoryStateMergeOperation::tryMapMemoryNodeIdToInput(
          callEntryMerge,
          memoryStateId);
      if (!entryMergeInput)
        throw std::runtime_error("MemoryStateId in call exit split not found in call entry merge");
      output.divert_users(entryMergeInput->origin());
    }
  }
}

/**
 * Finds alloca nodes from the callee that have been copied into the caller.
 * Moves the alloca nodes to the top level of the caller, and routes their outputs to their users.
 * This avoids having allocas inside theta nodes, which could otherwise cause the stack to overflow.
 * @param callee the function that has been inlined into the caller
 * @param caller the function that now has a copy of callee copied inside it
 * @param smap the substitution map used during copying
 */
static void
hoistInlinedAllocas(
    const rvsdg::LambdaNode & callee,
    rvsdg::LambdaNode & caller,
    rvsdg::SubstitutionMap & smap)
{
  // All alloca operations in the callee must be on the top level, with constant count,
  // otherwise the callee would not have qualified for being inlined

  for (auto & node : callee.subregion()->Nodes())
  {
    if (!is<AllocaOperation>(&node))
      continue;

    // Find the same alloca in the caller
    auto oldAllocaNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(smap.lookup(*node.output(0)));
    JLM_ASSERT(oldAllocaNode);

    auto countOrigin = AllocaOperation::getCountInput(*oldAllocaNode).origin();
    countOrigin = &rvsdg::traceOutputIntraProcedurally(*countOrigin);
    auto countNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*countOrigin);
    if (!countNode || countNode->ninputs() != 0)
      throw std::runtime_error("Alloca did not have a nullary count origin");

    // Create copies of the count node and alloca node at the top level
    const auto newCountNode = countNode->copy(caller.subregion(), {});
    const auto newAllocaNode = oldAllocaNode->copy(caller.subregion(), { newCountNode->output(0) });

    // Route the outputs of the new alloca to the region of the old alloca, and divert old users
    for (size_t n = 0; n < newAllocaNode->noutputs(); n++)
    {
      auto & oldOutput = *oldAllocaNode->output(n);
      auto & newOutput = *newAllocaNode->output(n);
      auto & routed = rvsdg::RouteToRegion(newOutput, *oldAllocaNode->region());
      oldOutput.divert_users(&routed);
    }

    // Remove the old alloca node, which is now dead
    remove(oldAllocaNode);
  }
}

void
FunctionInlining::inlineCall(
    rvsdg::SimpleNode & callNode,
    rvsdg::LambdaNode & caller,
    const rvsdg::LambdaNode & callee)
{
  JLM_ASSERT(is<CallOperation>(&callNode));

  // Make note of the call's entry and exit memory state nodes, if they exist
  auto callEntryMemoryStateMerge = CallOperation::tryGetMemoryStateEntryMerge(callNode);
  auto callExitMemoryStateMerge = CallOperation::tryGetMemoryStateExitSplit(callNode);

  // Set up substitution map for function arguments and context variables
  rvsdg::SubstitutionMap smap;
  auto arguments = callee.GetFunctionArguments();
  for (size_t n = 0; n < arguments.size(); n++)
  {
    smap.insert(arguments[n], callNode.input(n + 1)->origin());
  }

  const auto routedDeps = routeContextVariablesToRegion(*callNode.region(), callee);
  const auto contextVars = callee.GetContextVars();
  JLM_ASSERT(contextVars.size() == routedDeps.size());
  for (size_t n = 0; n < contextVars.size(); n++)
  {
    smap.insert(contextVars[n].inner, routedDeps[n]);
  }

  // Use the substitution map to copy the function body into the caller region
  callee.subregion()->copy(callNode.region(), smap, false, false);

  // Move all users of the call node's outputs to the callee's result origins
  const auto calleeResults = callee.GetFunctionResults();
  JLM_ASSERT(callNode.noutputs() == calleeResults.size());
  for (size_t n = 0; n < callNode.noutputs(); n++)
  {
    const auto resultOrigin = calleeResults[n]->origin();
    const auto newOrigin = smap.lookup(resultOrigin);
    JLM_ASSERT(newOrigin);
    callNode.output(n)->divert_users(newOrigin);
  }

  // If the callee was copied into a structural node within the caller function,
  // hoist any copied alloca nodes to the top level region of the caller function
  if (callNode.region() != caller.subregion())
  {
    hoistInlinedAllocas(callee, caller, smap);
  }

  // The call node is now dead. Remove it
  remove(&callNode);

  // If the call had memory state merge and split nodes,
  // try connecting memory state edges directly instead
  if (callEntryMemoryStateMerge && callExitMemoryStateMerge)
  {
    tryRerouteMemoryStateMergeAndSplit(*callEntryMemoryStateMerge, *callExitMemoryStateMerge);
  }
}

void
FunctionInlining::inlineCall(rvsdg::SimpleNode & callNode, const rvsdg::LambdaNode & callee)
{
  auto & caller = rvsdg::getSurroundingLambdaNode(callNode);
  inlineCall(callNode, caller, callee);
}

bool
FunctionInlining::canBeInlined(rvsdg::Region & region, bool topLevelRegion)
{
  for (auto & node : region.Nodes())
  {
    if (const auto structural = dynamic_cast<rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structural->Subregions())
      {
        if (!canBeInlined(subregion, false))
          return false;
      }
    }
    else if (is<AllocaOperation>(&node))
    {
      // Having allocas that are not on the top level of the function disqualifies from inlining
      if (!topLevelRegion)
        return false;

      // Having allocation sizes that are not compile time constants also disqualifies from inlining
      auto countOutput = AllocaOperation::getCountInput(node).origin();
      countOutput = &rvsdg::traceOutputIntraProcedurally(*countOutput);
      auto countNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*countOutput);

      // The count must come from a node, and it must be nullary
      if (!countNode || countNode->ninputs() != 0)
        return false;
    }
    else if (const auto [simple, callOp] =
                 rvsdg::TryGetSimpleNodeAndOptionalOp<CallOperation>(node);
             simple && callOp)
    {
      const auto classification = CallOperation::ClassifyCall(*simple);
      if (classification->isSetjmpCall())
      {
        // Calling setjmp weakens guarantees about local variables in the caller,
        // but not local variables in the caller's caller. Inlining would mix them up.
        return false;
      }
      if (classification->isVaStartCall())
      {
        // Calling va_start requires parameters to be passed in as expected by the ABI.
        // This gets broken if we start inlining.
        return false;
      }
    }
  }

  return true;
}

bool
FunctionInlining::canBeInlined(const rvsdg::LambdaNode & callee)
{
  return canBeInlined(*callee.subregion(), true);
}

bool
FunctionInlining::shouldInline(
    [[maybe_unused]] rvsdg::SimpleNode & callNode,
    [[maybe_unused]] rvsdg::LambdaNode & caller,
    rvsdg::LambdaNode & callee)
{
  // For now the inlining heuristic is very simple: Inline functions that are called exactly once
  return context_->functionsCalledOnce.Contains(&callee);
}

void
FunctionInlining::considerCallForInlining(
    rvsdg::SimpleNode & callNode,
    rvsdg::LambdaNode & callerLambda)
{
  context_->numFunctionCalls++;

  auto classification = CallOperation::ClassifyCall(callNode);
  if (!classification->IsDirectCall())
    return;

  auto & calleeOutput = classification->GetLambdaOutput();
  auto callee = rvsdg::TryGetOwnerNode<rvsdg::LambdaNode>(calleeOutput);
  JLM_ASSERT(callee != nullptr);

  // We can not inline a function into itself
  if (callee == &callerLambda)
    return;

  // We can only inline functions that have been marked as inlineable
  if (!context_->inlineableFunctions.Contains(callee))
    return;

  // At this point we know that it is technically possible to do inlining
  context_->numInlineableCalls++;
  if (shouldInline(callNode, callerLambda, *callee))
  {
    context_->numInlinedCalls++;
    inlineCall(callNode, callerLambda, *callee);
  }
}

void
FunctionInlining::visitIntraProceduralRegion(rvsdg::Region & region, rvsdg::LambdaNode & lambda)
{
  for (auto node : rvsdg::TopDownTraverser(&region))
  {
    rvsdg::MatchType(
        *node,
        [&](rvsdg::StructuralNode & structural)
        {
          for (auto & subregion : structural.Subregions())
          {
            visitIntraProceduralRegion(subregion, lambda);
          }
        },
        [&](rvsdg::SimpleNode & simple)
        {
          if (is<CallOperation>(&simple))
          {
            considerCallForInlining(simple, lambda);
          }
        });
  }
}

void
FunctionInlining::visitLambda(rvsdg::LambdaNode & lambda)
{
  context_->numFunctions++;

  // Visits the lambda's body and performs inlining of calls when determined to be beneficial
  visitIntraProceduralRegion(*lambda.subregion(), lambda);

  // After doing inlining inside lambda, we check if the function is eligible for being inlined
  if (canBeInlined(lambda))
    context_->inlineableFunctions.insert(&lambda);

  // Check if the function is only called once, and not exported from the module.
  // In which case inlining it is "free" in terms of total code size
  auto callSummary = ComputeCallSummary(lambda);
  if (callSummary.HasOnlyDirectCalls() && callSummary.NumDirectCalls() == 1)
    context_->functionsCalledOnce.insert(&lambda);
}

void
FunctionInlining::visitInterProceduralRegion(rvsdg::Region & region)
{
  for (auto node : rvsdg::TopDownTraverser(&region))
  {
    rvsdg::MatchType(
        *node,
        [&](rvsdg::PhiNode & phi)
        {
          visitInterProceduralRegion(*phi.subregion());
        },
        [&](rvsdg::LambdaNode & lambda)
        {
          visitLambda(lambda);
        });
  }
}

void
FunctionInlining::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::create(module.SourceFilePath().value_or(util::FilePath("")));

  context_ = std::make_unique<Context>();
  statistics->start();
  visitInterProceduralRegion(module.Rvsdg().GetRootRegion());
  statistics->stop(
      context_->numFunctions,
      context_->inlineableFunctions.Size(),
      context_->numFunctionCalls,
      context_->numInlineableCalls,
      context_->numInlinedCalls);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  context_.reset();
}

}

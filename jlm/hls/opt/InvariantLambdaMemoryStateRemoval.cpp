/*
 * Copyright 2025 Magnus Själander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/opt/InvariantLambdaMemoryStateRemoval.hpp>
#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::hls
{

void
InvariantLambdaMemoryStateRemoval::RemoveInvariantMemoryStateEdges(
    rvsdg::RegionResult & memoryStateResult)
{
  // We only apply this for memory state edges that is invariant between LambdaEntryMemoryStateSplit
  // and LambdaExitMemoryStateMerge nodes.
  auto exitNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*memoryStateResult.origin());
  if (exitNode == nullptr
      || !rvsdg::is<const llvm::LambdaExitMemoryStateMergeOperation>(exitNode->GetOperation()))
  {
    return;
  }

  // Check if we have any invariant edge(s) between the two nodes
  std::vector<size_t> nonInvariantIndices;
  std::vector<rvsdg::Output *> nonInvariantOutputs;
  rvsdg::Node * entryNode = nullptr;
  for (size_t i = 0; i < exitNode->ninputs(); i++)
  {
    // Check if the output has only one user and if it is a LambdaEntryMemoryStateMerge
    auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*exitNode->input(i)->origin());
    if (node != nullptr
        && jlm::rvsdg::is<const llvm::LambdaEntryMemoryStateSplitOperation>(node->GetOperation()))
    {
      // Found an invariant memory state edge, so going to replace the entryNode
      entryNode = node;
    }
    else
    {
      // Keep track of edges that is to be kept since they are not invariant
      nonInvariantOutputs.push_back(exitNode->input(i)->origin());
      // Also keep track of the index to be used for diverting edges
      nonInvariantIndices.push_back(i);
    }
  }

  // If the entryNode is not set, then we haven't found any invariant edges
  if (entryNode == nullptr)
  {
    return;
  }

  // Replace LambdaEntryMemoryStateSplit and LambdaExitMemoryStateMerge nodes
  if (nonInvariantOutputs.size() == 0)
  {
    // The memory state edge(s) are invariant, so we could in principle remove all of them from the
    // lambda. However, the LLVM dialect expects to always have a memory state, so we connect the
    // argument directly to the result
    memoryStateResult.divert_to(entryNode->input(0)->origin());
  }
  else if (nonInvariantOutputs.size() == 1)
  {
    // Single edge that is not invariant, so we can elmintate the two MemoryState nodes
    memoryStateResult.divert_to(nonInvariantOutputs[0]);
    entryNode->output(nonInvariantIndices[0])->divert_users(entryNode->input(0)->origin());
  }
  else
  {
    // Replace the entry and exit node with new ones without the invariant edge(s)
    auto newEntryNodeOutputs = llvm::LambdaEntryMemoryStateSplitOperation::Create(
        *entryNode->input(0)->origin(),
        nonInvariantIndices.size());
    memoryStateResult.divert_to(&llvm::LambdaExitMemoryStateMergeOperation::Create(
        *exitNode->region(),
        nonInvariantOutputs));
    int i = 0;
    for (auto index : nonInvariantIndices)
    {
      entryNode->output(index)->divert_users(newEntryNodeOutputs[i]);
      i++;
    }
  }
  JLM_ASSERT(exitNode->IsDead());
  rvsdg::remove(exitNode);
  JLM_ASSERT(entryNode->IsDead());
  rvsdg::remove(entryNode);
}

void
InvariantLambdaMemoryStateRemoval::RemoveInvariantLambdaMemoryStateEdges(
    rvsdg::RvsdgModule & rvsdgModule)
{
  for (auto & node : rvsdg::Graph::ExtractTailNodes(rvsdgModule.Rvsdg()))
  {
    if (auto lambda = dynamic_cast<const rvsdg::LambdaNode *>(node))
    {
      if (rvsdg::is<const llvm::LlvmLambdaOperation>(lambda->GetOperation())
          && lambda->output()->nusers() == 1
          && dynamic_cast<const jlm::rvsdg::GraphExport *>(*lambda->output()->begin()))
      {
        RemoveInvariantMemoryStateEdges(
            *dynamic_cast<rvsdg::RegionResult *>(&llvm::GetMemoryStateRegionResult(*lambda)));
      }
    }
  }
}

class InvariantLambdaMemoryStateRemoval::Statistics final : public util::Statistics
{
public:
  ~Statistics() noexcept override = default;

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

InvariantLambdaMemoryStateRemoval::~InvariantLambdaMemoryStateRemoval() noexcept = default;

void
InvariantLambdaMemoryStateRemoval::Run(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFilePath().value());

  statistics->Start();
  RemoveInvariantLambdaMemoryStateEdges(rvsdgModule);
  statistics->Stop();

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

void
InvariantLambdaMemoryStateRemoval::CreateAndRun(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  hls::InvariantLambdaMemoryStateRemoval memStateRemoval;
  memStateRemoval.Run(rvsdgModule, statisticsCollector);
}

} // namespace jlm::hls

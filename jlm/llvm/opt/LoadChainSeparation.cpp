/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/opt/LoadChainSeparation.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <RVSDG/Ops.h.inc>

namespace jlm::llvm
{

struct ChainLink
{
  virtual ~ChainLink() noexcept = default;
};

struct LoadChainLink final : ChainLink
{
  ~LoadChainLink() noexcept override = default;

  LoadChainLink(rvsdg::Output & output, std::shared_ptr<ChainLink> nextLink)
      : output(output),
        nextLink(std::move(nextLink))

  {
    JLM_ASSERT(rvsdg::is<MemoryStateType>(output.Type()));
    JLM_ASSERT(is<LoadOperation>(rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output)));
  }

  static std::shared_ptr<ChainLink>
  Create(rvsdg::Output & output, std::shared_ptr<ChainLink> nextLink)
  {
    return std::make_shared<LoadChainLink>(output, std::move(nextLink));
  }

  rvsdg::Output & output;
  std::shared_ptr<ChainLink> nextLink;
};

struct ChainLinkEnd final : ChainLink
{
  ~ChainLinkEnd() noexcept override = default;

  explicit ChainLinkEnd(rvsdg::Output & output)
      : output(output)
  {
    JLM_ASSERT(rvsdg::is<MemoryStateType>(output.Type()));
    JLM_ASSERT(!is<LoadOperation>(rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output)));
  }

  static std::shared_ptr<ChainLink>
  Create(rvsdg::Output & output)
  {
    return std::make_shared<ChainLinkEnd>(output);
  }

  rvsdg::Output & output;
};

struct GammaChainLink final : ChainLink
{
  ~GammaChainLink() noexcept override = default;

  explicit GammaChainLink(std::vector<std::shared_ptr<ChainLink>> subregionChainLinks)
      : subregionChainLinks(std::move(subregionChainLinks))
  {}

  static std::shared_ptr<ChainLink>
  Create(std::vector<std::shared_ptr<ChainLink>> subregionChainLinks)
  {
    return std::make_shared<GammaChainLink>(std::move(subregionChainLinks));
  }

  bool
  hasSingleSubregionChainEnd() const noexcept
  {
    JLM_ASSERT(!subregionChainLinks.empty());
    auto & chainLink = subregionChainLinks[0];

    for ()
  }

  std::vector<std::shared_ptr<ChainLink>> subregionChainLinks{};
};

static std::unordered_map<const LoadChainLink *, rvsdg::Output *>
createNewLoadMemoryStateOperands(const ChainLink & chainLink)
{
  using LoadStateOperandMap = std::unordered_map<const LoadChainLink *, rvsdg::Output *>;

  std::function<rvsdg::Output *(
      const ChainLink &,
      std::unordered_map<const LoadChainLink *, rvsdg::Output *> &)>
      createOperands = [&](const ChainLink & chainLink,
                           std::unordered_map<const LoadChainLink *, rvsdg::Output *> & operandMap)
  {
    return rvsdg::MatchTypeOrFail(
        chainLink,
        [&](const ChainLinkEnd & chainLinkEnd)
        {
          return &chainLinkEnd.output;
        },
        [&](const LoadChainLink & loadChainLink)
        {
          const auto newOperand = createOperands(*loadChainLink.nextLink, operandMap);
          operandMap[&loadChainLink] = newOperand;
          return newOperand;
        },
        [&](const GammaChainLink & gammaChainLink)
        {
          JLM_ASSERT(0);
          return nullptr;
        });
  };

  LoadStateOperandMap operandMap;
  createOperands(chainLink, operandMap);
  return operandMap;
}

std::vector<rvsdg::Output *> static computeJoinOperands(const ChainLink & chainLink)
{
  std::function<void(const ChainLink &, std::vector<rvsdg::Output *> &)> compute =
      [&](const ChainLink & chainLink, std::vector<rvsdg::Output *> & joinOperands)
  {
    rvsdg::MatchTypeOrFail(
        chainLink,
        [](const ChainLinkEnd &)
        {
          // Nothing needs to be done
        },
        [&](const LoadChainLink & loadChainLink)
        {
          joinOperands.push_back(&loadChainLink.output);
          compute(*loadChainLink.nextLink, joinOperands);
        });
  };

  std::vector<rvsdg::Output *> joinOperands;
  compute(chainLink, joinOperands);
  return joinOperands;
}

LoadChainSeparation::~LoadChainSeparation() noexcept = default;

LoadChainSeparation::LoadChainSeparation()
    : Transformation("LoadChainSeparation")
{}

void
LoadChainSeparation::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector &)
{
  util::HashSet<rvsdg::Output *> loadChainStarts;
  findLoadChainStarts(module.Rvsdg().GetRootRegion(), loadChainStarts);

  for (auto & memoryStateOutput : loadChainStarts.Items())
  {
    auto loadChain = traceLoadChain(*memoryStateOutput);
    separateLoadChain(*loadChain);
  }
}

void
LoadChainSeparation::findLoadChainStarts(
    rvsdg::Region & region,
    util::HashSet<rvsdg::Output *> & loadChainEnds)
{
  for (auto & node : region.Nodes())
  {
    // Handle innermost regions first
    if (const auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        findLoadChainStarts(subregion, loadChainEnds);
      }
    }

    if (!rvsdg::is<LoadOperation>(&node))
    {
      continue;
    }

    for (auto & memoryStateOutput : LoadOperation::MemoryStateOutputs(node))
    {
      if (hasLoadNodeAsUserOwner(memoryStateOutput))
      {
        continue;
      }

      auto & memoryStateInput = LoadOperation::MapMemoryStateOutputToInput(memoryStateOutput);
      if (hasLoadNodeAsOperandOwner(memoryStateInput))
      {
        loadChainEnds.insert(&memoryStateOutput);
      }
    }
  }
}

void
LoadChainSeparation::separateLoadChain(const LoadChainLink & loadChainLink)
{
  auto operandMap = createNewLoadMemoryStateOperands(loadChainLink);
  auto joinOperands = computeJoinOperands(loadChainLink);
  JLM_ASSERT(joinOperands.size() > 1);

  // Divert the operands of the respective inputs for each encountered load
  for (auto [loadChainLink, newOperand] : operandMap)
  {
    auto & memoryStateInput = LoadOperation::MapMemoryStateOutputToInput(loadChainLink->output);
    memoryStateInput.divert_to(newOperand);
  }

  // Create join node and divert the current memory state output
  auto & joinNode = MemoryStateJoinOperation::CreateNode(joinOperands);
  loadChainLink.output.divertUsersWhere(
      *joinNode.output(0),
      [&joinNode](const rvsdg::Input & user)
      {
        return rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(user) != &joinNode;
      });
}

std::shared_ptr<const LoadChainLink>
LoadChainSeparation::traceLoadChain(rvsdg::Output & output)
{
  using ChainLinkMap = std::unordered_map<rvsdg::Output *, std::shared_ptr<ChainLink>>;

  std::function<std::shared_ptr<ChainLink>(rvsdg::Output &, ChainLinkMap &)> trace =
      [&](rvsdg::Output & output, ChainLinkMap & chainLinkMap)
  {
    JLM_ASSERT(rvsdg::is<MemoryStateType>(output.Type()));

    // If we have seen this output before, just return the chain link
    if (const auto it = chainLinkMap.find(&output); it != chainLinkMap.end())
    {
      return it->second;
    }

    // Handle gamma subregion arguments
    if (const auto gammaNode = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(output))
    {
      const auto roleVar = gammaNode->MapBranchArgument(output);
      if (const auto entryVar = std::get_if<rvsdg::GammaNode::EntryVar>(&roleVar))
      {
        const auto argument = entryVar->input->origin();
        auto chainLink = trace(*argument, chainLinkMap);
        chainLinkMap[argument] = chainLink;
        return chainLink;
      }

      // We should never end up here
      throw std::logic_error("Unsupported gamma role");
    }

    // Handle gamma outputs
    if (const auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(output))
    {
      std::vector<std::shared_ptr<ChainLink>> subregionChainLinks;
      auto [branchResults, _] = gammaNode->MapOutputExitVar(output);
      for (const auto branchResult : branchResults)
      {
        auto chainLink = trace(*branchResult->origin(), chainLinkMap);
        subregionChainLinks.push_back(chainLink);
      }

      auto chainLink = GammaChainLink::Create(std::move(subregionChainLinks));
      chainLinkMap[&output] = chainLink;
      return chainLink;
    }

    if (is<LoadOperation>(rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output)))
    {
      auto nextLink =
          trace(*LoadOperation::MapMemoryStateOutputToInput(output).origin(), chainLinkMap);
      const auto chainLink = LoadChainLink::Create(output, nextLink);
      chainLinkMap[&output] = chainLink;
      return chainLink;
    }

    auto chainLink = ChainLinkEnd::Create(output);
    chainLinkMap[&output] = chainLink;
    return chainLink;
  };

  std::unordered_map<rvsdg::Output *, std::shared_ptr<ChainLink>> chainLinkMap;
  const auto chainLink = trace(output, chainLinkMap);

  JLM_ASSERT(std::dynamic_pointer_cast<const LoadChainLink>(chainLink));
  return std::static_pointer_cast<const LoadChainLink>(chainLink);
}

bool
LoadChainSeparation::hasLoadNodeAsOperandOwner(const rvsdg::Input & input)
{
  const auto & operand = *input.origin();

  if (rvsdg::IsOwnerNodeOperation<LoadOperation>(operand))
    return true;

  // Handle gamma outputs
  if (const auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(operand))
  {
    auto [branchResults, _] = gammaNode->MapOutputExitVar(operand);
    for (const auto branchResult : branchResults)
    {
      if (hasLoadNodeAsOperandOwner(*branchResult))
        return true;
    }

    return false;
  }

  // Handle gamma subregion arguments
  if (const auto gammaNode = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(operand))
  {
    const auto roleVar = gammaNode->MapBranchArgument(operand);
    if (const auto entryVar = std::get_if<rvsdg::GammaNode::EntryVar>(&roleVar))
    {
      return hasLoadNodeAsOperandOwner(*entryVar->input);
    }

    return false;
  }

  return false;
}

bool
LoadChainSeparation::hasLoadNodeAsUserOwner(const rvsdg::Output & output)
{
  for (auto & user : output.Users())
  {
    if (rvsdg::IsOwnerNodeOperation<LoadOperation>(user))
      return true;

    // Handle gamma inputs
    if (const auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(user))
    {
      const auto roleVar = gammaNode->MapInput(user);
      if (const auto entryVar = std::get_if<rvsdg::GammaNode::EntryVar>(&roleVar))
      {
        for (const auto branchArgument : entryVar->branchArgument)
        {
          if (hasLoadNodeAsUserOwner(*branchArgument))
          {
            return true;
          }
        }
      }

      return false;
    }

    // Handle gamma subregion results
    if (const auto gammaNode = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(user))
    {
      const auto [_, gammaOutput] = gammaNode->MapBranchResultExitVar(user);
      return hasLoadNodeAsUserOwner(*gammaOutput);
    }
  }

  return false;
}

}

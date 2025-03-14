/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/backend/RegionSequentializer.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/util/HashSet.hpp>

namespace jlm::llvm
{

RegionSequentializer::~RegionSequentializer() noexcept = default;

ExhaustiveSingleRegionSequentializer::~ExhaustiveSingleRegionSequentializer() noexcept = default;

ExhaustiveSingleRegionSequentializer::ExhaustiveSingleRegionSequentializer(
    const rvsdg::Region & region)
    : Region_(&region)
{
  ComputeSequentializations(region);
}

void
ExhaustiveSingleRegionSequentializer::ComputeSequentializations(const rvsdg::Region & region)
{
  util::HashSet<const rvsdg::Node *> visited;
  std::vector<const rvsdg::Node *> sequentializedNodes;
  ComputeSequentializations(region, visited, sequentializedNodes);
}

void
ExhaustiveSingleRegionSequentializer::ComputeSequentializations(
    const rvsdg::Region & region,
    util::HashSet<const rvsdg::Node *> & visited,
    std::vector<const rvsdg::Node *> & sequentializedNodes) noexcept
{
  for (auto & node : region.Nodes())
  {
    if (AllPredecessorsVisited(node, visited) && !visited.Contains(&node))
    {
      sequentializedNodes.push_back(&node);
      visited.Insert(&node);

      ComputeSequentializations(region, visited, sequentializedNodes);

      visited.Remove(&node);
      sequentializedNodes.pop_back();
    }
  }

  if (sequentializedNodes.size() == region.nnodes())
  {
    Sequentializations_.emplace_back(sequentializedNodes);
  }
}

std::optional<SequentializationMap>
ExhaustiveSingleRegionSequentializer::ComputeNextSequentialization()
{
  if (!HasMoreSequentializations())
    return std::nullopt;

  const auto sequentialization = Sequentializations_[CurrentSequentialization_];
  CurrentSequentialization_++;

  SequentializationMap sequentializationMap;
  sequentializationMap[Region_] = sequentialization;
  return sequentializationMap;
}

void
ExhaustiveSingleRegionSequentializer::Reset()
{
  CurrentSequentialization_ = 0;
}

bool
ExhaustiveSingleRegionSequentializer::AllPredecessorsVisited(
    const rvsdg::Node & node,
    const util::HashSet<const rvsdg::Node *> & visited)
{
  for (size_t n = 0; n < node.ninputs(); n++)
  {
    auto & origin = *node.input(n)->origin();
    if (const auto predecessor = rvsdg::TryGetOwnerNode<rvsdg::Node>(origin))
    {
      if (!visited.Contains(predecessor))
      {
        return false;
      }
    }
  }

  return true;
}

ExhaustiveRegionSequentializer::~ExhaustiveRegionSequentializer() noexcept = default;

ExhaustiveRegionSequentializer::ExhaustiveRegionSequentializer(const rvsdg::Region & region)
{
  InitializeSequentializers(region);
}

void
ExhaustiveRegionSequentializer::InitializeSequentializers(const rvsdg::Region & region)
{
  Sequentializers_[&region] = std::make_unique<ExhaustiveSingleRegionSequentializer>(region);
  for (auto & node : region.Nodes())
  {
    if (const auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (size_t n = 0; n < structuralNode->nsubregions(); n++)
      {
        const auto subregion = structuralNode->subregion(n);
        InitializeSequentializers(*subregion);
      }
    }
  }
}

std::optional<SequentializationMap>
ExhaustiveRegionSequentializer::ComputeNextSequentialization()
{
  if (!HasMoreSequentializations())
    return std::nullopt;

  SequentializationMap sequentializationMap;
  for (auto & [region, sequentializer] : Sequentializers_)
  {
    auto sequentialization = sequentializer->ComputeNextSequentialization();
    if (!sequentialization.has_value())
    {
      sequentializer->Reset();
      sequentialization = sequentializer->ComputeNextSequentialization();
    }

    sequentializationMap[region] = std::move(sequentialization.value()[region]);
  }

  return sequentializationMap;
}

bool
ExhaustiveRegionSequentializer::HasMoreSequentializations() const noexcept
{
  for (auto & [_, sequentializer] : Sequentializers_)
  {
    if (sequentializer->HasMoreSequentializations())
      return true;
  }

  return false;
}

}

/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/backend/RegionSequentializer.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/HashSet.hpp>

namespace jlm::llvm
{

RegionSequentializer::~RegionSequentializer() noexcept = default;

ExhaustiveSingleRegionSequentializer::~ExhaustiveSingleRegionSequentializer() noexcept = default;

ExhaustiveSingleRegionSequentializer::ExhaustiveSingleRegionSequentializer() = default;

void
ExhaustiveSingleRegionSequentializer::Initialize(rvsdg::Region & region)
{
  Sequentializations_ = std::vector<Sequentialization>();

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
    Sequentializations_.value().emplace_back(sequentializedNodes);
  }
}

void
ExhaustiveSingleRegionSequentializer::ComputeNextSequentialization()
{
  if (!HasMoreSequentializations())
    return;

  CurrentSequentialization_++;
}

std::optional<SequentializationMap>
ExhaustiveSingleRegionSequentializer::GetSequentializations() const
{
  if (!HasMoreSequentializations())
    return std::nullopt;

  if (!Sequentializations_.has_value())
    return std::nullopt;

  const auto sequentialization = Sequentializations_.value()[CurrentSequentialization_];

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

ExhaustiveRegionSequentializer::ExhaustiveRegionSequentializer(rvsdg::Region & region)
{
  InitializeSequentializers(region);
}

void
ExhaustiveRegionSequentializer::InitializeSequentializers(rvsdg::Region & region)
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

void
ExhaustiveRegionSequentializer::ComputeNextSequentialization()
{
  if (!HasMoreSequentializations())
  {
    CurrentSequentializations_ = std::nullopt;
    return;
  }

  for (auto & [region, sequentializer] : Sequentializers_)
  {
    auto sequentialization = sequentializer->ComputeNextSequentialization();
    if (!sequentialization.has_value())
    {
      sequentializer->Reset();
      sequentialization = sequentializer->ComputeNextSequentialization();
    }

    CurrentSequentializations_[region] = std::move(sequentialization.value()[region]);
  }
}

std::optional<SequentializationMap>
ExhaustiveRegionSequentializer::GetSequentializations() const
{
  return CurrentSequentializations_;
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

IdempotentRegionSequentializer::~IdempotentRegionSequentializer() noexcept = default;

IdempotentRegionSequentializer::IdempotentRegionSequentializer(rvsdg::Region & region)
    : Region_(&region)
{
  for (const auto & node : rvsdg::TopDownTraverser(&region))
  {
    Sequentialization_.push_back(node);
  }
}

void
IdempotentRegionSequentializer::ComputeNextSequentialization()
{}

std::optional<SequentializationMap>
IdempotentRegionSequentializer::GetSequentializations() const
{
  return Sequentialization_;
}

}

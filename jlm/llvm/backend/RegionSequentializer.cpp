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

RegionSequentializer::~RegionSequentializer() = default;

RegionSequentializer::RegionSequentializer(rvsdg::Region & region)
    : Region_(&region)
{}

ExhaustiveRegionSequentializer::ExhaustiveRegionSequentializer(rvsdg::Region & region)
    : RegionSequentializer(region)
{
  util::HashSet<const rvsdg::Node *> visited;
  std::vector<const rvsdg::Node *> sequentializedNodes;
  ComputeSequentializations(region, visited, sequentializedNodes);
}

void
ExhaustiveRegionSequentializer::ComputeNextSequentialization()
{
  if (HasMoreSequentializations())
    CurrentSequentialization_++;
  else
    CurrentSequentialization_ = 0;
}

Sequentialization
ExhaustiveRegionSequentializer::GetSequentialization()
{
  if (!HasMoreSequentializations())
    ComputeNextSequentialization();

  return Sequentializations_[CurrentSequentialization_];
}

bool
ExhaustiveRegionSequentializer::HasMoreSequentializations() const noexcept
{
  JLM_ASSERT(CurrentSequentialization_ <= Sequentializations_.size());
  return CurrentSequentialization_ != Sequentializations_.size();
}

void
ExhaustiveRegionSequentializer::ComputeSequentializations(
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

bool
ExhaustiveRegionSequentializer::AllPredecessorsVisited(
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

IdempotentRegionSequentializer::~IdempotentRegionSequentializer() noexcept = default;

IdempotentRegionSequentializer::IdempotentRegionSequentializer(rvsdg::Region & region)
    : RegionSequentializer(region)
{}

void
IdempotentRegionSequentializer::ComputeNextSequentialization()
{}

Sequentialization
IdempotentRegionSequentializer::GetSequentialization()
{
  auto sequentialization = Sequentialization();
  for (const auto node : rvsdg::TopDownTraverser(&GetRegion()))
  {
    sequentialization.push_back(node);
  }
  return sequentialization;
}

bool
IdempotentRegionSequentializer::HasMoreSequentializations() const noexcept
{
  return true;
}

RegionTreeSequentializer::~RegionTreeSequentializer() noexcept = default;

RegionTreeSequentializer::RegionTreeSequentializer(SequentializerMap sequentializerMap)
    : Sequentializers_(std::move(sequentializerMap))
{}

void
RegionTreeSequentializer::ComputeNextSequentializations()
{
  for (auto & [region, sequentializer] : Sequentializers_)
  {
    sequentializer->ComputeNextSequentialization();
  }
}

bool
RegionTreeSequentializer::HasMoreSequentializations() const noexcept
{
  for (auto & [_, sequentializer] : Sequentializers_)
  {
    if (sequentializer->HasMoreSequentializations())
      return true;
  }

  return false;
}

Sequentialization
RegionTreeSequentializer::GetSequentialization(rvsdg::Region & region)
{
  JLM_ASSERT(Sequentializers_.find(&region) != Sequentializers_.end());
  return Sequentializers_[&region]->GetSequentialization();
}

template<class TRegionSequentializer>
void
InitializeUniformRegionTreeSequentializer(
    rvsdg::Region & region,
    SequentializerMap & sequentializerMap)
{
  auto sequentializer = std::make_unique<TRegionSequentializer>(region);

  for (auto & node : region.Nodes())
  {
    if (const auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (size_t n = 0; n < structuralNode->nsubregions(); n++)
      {
        const auto subregion = structuralNode->subregion(n);
        InitializeUniformRegionTreeSequentializer<TRegionSequentializer>(
            *subregion,
            sequentializerMap);
      }
    }
  }

  sequentializerMap[&region] = std::move(sequentializer);
}

RegionTreeSequentializer
CreateIdempotentRegionTreeSequentializer(rvsdg::Region & rootRegion)
{
  SequentializerMap sequentializerMap;
  InitializeUniformRegionTreeSequentializer<IdempotentRegionSequentializer>(
      rootRegion,
      sequentializerMap);
  return RegionTreeSequentializer(std::move(sequentializerMap));
}

RegionTreeSequentializer
CreateExhaustiveRegionTreeSequentializer(rvsdg::Region & rootRegion)
{
  SequentializerMap sequentializerMap;
  InitializeUniformRegionTreeSequentializer<ExhaustiveRegionSequentializer>(
      rootRegion,
      sequentializerMap);
  return RegionTreeSequentializer(std::move(sequentializerMap));
}

}

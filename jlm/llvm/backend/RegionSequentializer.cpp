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

// FIXME: add documentation
class ExhaustiveSingleRegionSequentializer final : public RegionSequentializer
{
public:
  ~ExhaustiveSingleRegionSequentializer() noexcept override = default;

  ExhaustiveSingleRegionSequentializer() = default;

  void
  Initialize(rvsdg::Region & region) override
  {
    Sequentializations_ = std::vector<Sequentialization>();

    util::HashSet<const rvsdg::Node *> visited;
    std::vector<const rvsdg::Node *> sequentializedNodes;
    ComputeSequentializations(region, visited, sequentializedNodes);
  }

  void
  ComputeNextSequentializations() override
  {
    if (HasMoreSequentializations())
      CurrentSequentialization_++;
    else
      CurrentSequentialization_ = 0;
  }

  SequentializationMap
  GetSequentializations() override
  {
    if (!HasMoreSequentializations())
      CurrentSequentialization_ = 0;

    const auto sequentialization = Sequentializations_[CurrentSequentialization_];

    SequentializationMap sequentializationMap;
    sequentializationMap[Region_] = sequentialization;
    return sequentializationMap;
  }

  bool
  HasMoreSequentializations() const noexcept override
  {
    JLM_ASSERT(CurrentSequentialization_ < Sequentializations_.size());
    return (CurrentSequentialization_ + 1) != Sequentializations_.size();
  }

private:
  void
  ComputeSequentializations(
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

  static bool
  AllPredecessorsVisited(
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

  rvsdg::Region * Region_;
  size_t CurrentSequentialization_ = 0;
  std::vector<Sequentialization> Sequentializations_;
};

ExhaustiveRegionSequentializer::~ExhaustiveRegionSequentializer() noexcept = default;

ExhaustiveRegionSequentializer::ExhaustiveRegionSequentializer() = default;

void
ExhaustiveRegionSequentializer::Initialize(rvsdg::Region & region)
{
  auto sequentializer = std::make_unique<ExhaustiveSingleRegionSequentializer>();
  sequentializer->Initialize(region);

  for (auto & node : region.Nodes())
  {
    if (const auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (size_t n = 0; n < structuralNode->nsubregions(); n++)
      {
        const auto subregion = structuralNode->subregion(n);
        Initialize(*subregion);
      }
    }
  }

  Sequentializers_[&region] = std::move(sequentializer);
}

void
ExhaustiveRegionSequentializer::ComputeNextSequentializations()
{
  for (auto & [region, sequentializer] : Sequentializers_)
  {
    sequentializer->ComputeNextSequentializations();
    CurrentSequentializations_[region] = std::move(sequentializer->GetSequentializations()[region]);
  }
}

SequentializationMap
ExhaustiveRegionSequentializer::GetSequentializations()
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

IdempotentRegionSequentializer::IdempotentRegionSequentializer() = default;

void
IdempotentRegionSequentializer::Initialize(rvsdg::Region & region)
{
  auto sequentialization = Sequentialization();
  for (const auto & node : rvsdg::TopDownTraverser(&region))
  {
    sequentialization.push_back(node);
    if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(node))
    {
      for (size_t n = 0; n < structuralNode->nsubregions(); n++)
      {
        Initialize(*structuralNode->subregion(n));
      }
    }
  }
  Sequentialization_[&region] = std::move(sequentialization);
}

void
IdempotentRegionSequentializer::ComputeNextSequentializations()
{}

SequentializationMap
IdempotentRegionSequentializer::GetSequentializations()
{
  return Sequentialization_;
}

bool
IdempotentRegionSequentializer::HasMoreSequentializations() const noexcept
{
  return true;
}

}

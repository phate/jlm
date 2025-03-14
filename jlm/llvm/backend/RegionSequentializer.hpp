/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_BACKEND_REGIONSEQUENTIALIZER_HPP
#define JLM_LLVM_BACKEND_REGIONSEQUENTIALIZER_HPP

#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/util/HashSet.hpp>
#include <vector>

namespace jlm::rvsdg
{
class Node;
class Region;
}

namespace jlm::llvm
{

using Sequentialization = std::vector<const rvsdg::Node *>;
using SequentializationMap = std::unordered_map<const rvsdg::Region *, Sequentialization>;

// FIXME: add documentation
class RegionSequentializer
{
public:
  virtual ~RegionSequentializer() noexcept;

  virtual std::optional<SequentializationMap>
  ComputeNextSequentialization() = 0;
};

// FIXME: add documentation
class ExhaustiveSingleRegionSequentializer final : public RegionSequentializer
{
public:
  ~ExhaustiveSingleRegionSequentializer() noexcept override;

  explicit ExhaustiveSingleRegionSequentializer(const rvsdg::Region & region);

  std::optional<SequentializationMap>
  ComputeNextSequentialization() override;

  void
  Reset();

  bool
  HasMoreSequentializations() const noexcept
  {
    return CurrentSequentialization_ < Sequentializations_.size();
  }

private:
  void
  ComputeSequentializations(const rvsdg::Region & region);

  void
  ComputeSequentializations(
      const rvsdg::Region & region,
      util::HashSet<const rvsdg::Node *> & visited,
      std::vector<const rvsdg::Node *> & sequentializedNodes) noexcept;

  static bool
  AllPredecessorsVisited(
      const rvsdg::Node & node,
      const util::HashSet<const rvsdg::Node *> & visited);

  const rvsdg::Region * Region_;
  size_t CurrentSequentialization_ = 0;
  std::vector<Sequentialization> Sequentializations_;
};

// FIXME: add documentation
class ExhaustiveRegionSequentializer final : public RegionSequentializer
{
public:
  ~ExhaustiveRegionSequentializer() noexcept override;

  explicit ExhaustiveRegionSequentializer(const rvsdg::Region & region);

  bool
  HasMoreSequentializations() const noexcept;

  std::optional<SequentializationMap>
  ComputeNextSequentialization() override;

private:
  void
  InitializeSequentializers(const rvsdg::Region & region);

  std::unordered_map<const rvsdg::Region *, std::unique_ptr<ExhaustiveSingleRegionSequentializer>>
      Sequentializers_;
};

}

#endif // JLM_LLVM_BACKEND_REGIONSEQUENTIALIZER_HPP

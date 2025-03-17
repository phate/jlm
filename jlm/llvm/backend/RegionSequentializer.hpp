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
using SequentializationMap = std::unordered_map<rvsdg::Region *, Sequentialization>;

// FIXME: add documentation
class RegionSequentializer
{
public:
  virtual ~RegionSequentializer() noexcept;

  virtual void
  Initialize(rvsdg::Region & region) = 0;

  virtual void
  ComputeNextSequentialization() = 0;

  virtual std::optional<SequentializationMap>
  GetSequentializations() const = 0;
};

// FIXME: add documentation
class ExhaustiveSingleRegionSequentializer final : public RegionSequentializer
{
public:
  ~ExhaustiveSingleRegionSequentializer() noexcept override;

  ExhaustiveSingleRegionSequentializer();

  void
  Initialize(rvsdg::Region & region) override;

  void
  ComputeNextSequentialization() override;

  std::optional<SequentializationMap>
  GetSequentializations() const override;

  void
  Reset();

  bool
  HasMoreSequentializations() const noexcept
  {
    return CurrentSequentialization_ < Sequentializations_.value().size();
  }

private:
  void
  ComputeSequentializations(
      const rvsdg::Region & region,
      util::HashSet<const rvsdg::Node *> & visited,
      std::vector<const rvsdg::Node *> & sequentializedNodes) noexcept;

  static bool
  AllPredecessorsVisited(
      const rvsdg::Node & node,
      const util::HashSet<const rvsdg::Node *> & visited);

  rvsdg::Region * Region_;
  size_t CurrentSequentialization_ = 0;
  std::optional<std::vector<Sequentialization>> Sequentializations_;
};

// FIXME: add documentation
class ExhaustiveRegionSequentializer final : public RegionSequentializer
{
public:
  ~ExhaustiveRegionSequentializer() noexcept override;

  explicit ExhaustiveRegionSequentializer(rvsdg::Region & region);

  bool
  HasMoreSequentializations() const noexcept;

  void
  ComputeNextSequentialization() override;

  std::optional<SequentializationMap>
  GetSequentializations() const override;

private:
  void
  InitializeSequentializers(rvsdg::Region & topRegion);

  std::optional<SequentializationMap> CurrentSequentializations_;
  std::unordered_map<rvsdg::Region *, std::unique_ptr<ExhaustiveSingleRegionSequentializer>>
      Sequentializers_;
};

// FIXME: add documentation
class IdempotentRegionSequentializer final : public RegionSequentializer
{
public:
  ~IdempotentRegionSequentializer() noexcept override;

  explicit IdempotentRegionSequentializer(rvsdg::Region & region);

  void
  ComputeNextSequentialization() override;

  std::optional<SequentializationMap>
  GetSequentializations() const override;

private:
  rvsdg::Region * Region_;
  SequentializationMap Sequentialization_;
};

}

#endif // JLM_LLVM_BACKEND_REGIONSEQUENTIALIZER_HPP

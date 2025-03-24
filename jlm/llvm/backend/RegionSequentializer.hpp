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
  ComputeNextSequentializations() = 0;

  virtual SequentializationMap
  GetSequentializations() = 0;

  virtual bool
  HasMoreSequentializations() const noexcept = 0;
};

class ExhaustiveSingleRegionSequentializer;

// FIXME: add documentation
class ExhaustiveRegionSequentializer final : public RegionSequentializer
{
public:
  ~ExhaustiveRegionSequentializer() noexcept override;

  ExhaustiveRegionSequentializer();

  void
  Initialize(rvsdg::Region & region) override;

  void
  ComputeNextSequentializations() override;

  SequentializationMap
  GetSequentializations() override;

  bool
  HasMoreSequentializations() const noexcept override;

private:
  void
  InitializeSequentializers(rvsdg::Region & topRegion);

  SequentializationMap CurrentSequentializations_;
  std::unordered_map<rvsdg::Region *, std::unique_ptr<ExhaustiveSingleRegionSequentializer>>
      Sequentializers_;
};

// FIXME: add documentation
class IdempotentRegionSequentializer final : public RegionSequentializer
{
public:
  ~IdempotentRegionSequentializer() noexcept override;

  IdempotentRegionSequentializer();

  void
  Initialize(rvsdg::Region & region) override;

  void
  ComputeNextSequentializations() override;

  SequentializationMap
  GetSequentializations() override;

  bool
  HasMoreSequentializations() const noexcept override;

private:
  rvsdg::Region * Region_;
  SequentializationMap Sequentialization_;
};

}

#endif // JLM_LLVM_BACKEND_REGIONSEQUENTIALIZER_HPP

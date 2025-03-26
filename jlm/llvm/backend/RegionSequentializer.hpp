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

/**
 * The RegionSequentializer class computes sequentializations, i.e., topological orderings, for a
 * <b>single</b> region. The interface permits the computation of different sequentializations
 * by invoking ComputeNextSequentialization(). It can be checked whether there are still orderings
 * available by invoking HasMoreSequentializations().
 */
class RegionSequentializer
{
public:
  virtual ~RegionSequentializer() noexcept;

  explicit RegionSequentializer(rvsdg::Region & region);

  RegionSequentializer(const RegionSequentializer &) = delete;

  RegionSequentializer(RegionSequentializer &&) = delete;

  RegionSequentializer &
  operator=(const RegionSequentializer &) = delete;

  RegionSequentializer &
  operator=(RegionSequentializer &&) = delete;

  /**
   * @return The current sequentialization of the region. Nodes are stored in dependency-order in
   * the sequentialization, i.e., nodes without any dependency appear first in the
   * sequentialization.
   */
  virtual Sequentialization
  GetSequentialization() = 0;

  /**
   * @return True, if there are more sequentializations to return, otherwise false.
   */
  virtual bool
  HasMoreSequentializations() const noexcept = 0;

  /**
   * Computes the next sequentialization of the region.
   */
  virtual void
  ComputeNextSequentialization() = 0;

  /**
   * @return The region for which sequentializations are computed.
   */
  rvsdg::Region &
  GetRegion() const noexcept
  {
    return *Region_;
  }

private:
  rvsdg::Region * Region_;
};

/**
 * The ExhaustiveRegionSequentializer computes <b>all</b> sequentializations of a region. It is
 * possible to iterate through these sequentializations as follows:
 *
 * \code{.c}
 * while(sequentializer.HasMoreSequentializations)
 * {
 *   auto sequentialization = sequentializer.GetSequentialization();
 *   ...
 *   sequentializer.ComputeNextSequentialization();
 * }
 * \endcode
 *
 * The sequentializer will wrap around after the "last" sequentialization and
 * return the "first" sequentialization again.
 */
class ExhaustiveRegionSequentializer final : public RegionSequentializer
{
public:
  ~ExhaustiveRegionSequentializer() noexcept override = default;

  explicit ExhaustiveRegionSequentializer(rvsdg::Region & region);

  void
  ComputeNextSequentialization() override;

  Sequentialization
  GetSequentialization() override;

  bool
  HasMoreSequentializations() const noexcept override;

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

  size_t CurrentSequentialization_ = 0;
  std::vector<Sequentialization> Sequentializations_;
};

/**
 * The IdempotentRegionSequentializer computes a <b>single</b> sequentialization of a region and
 * only ever returns this single sequentialization.
 */
class IdempotentRegionSequentializer final : public RegionSequentializer
{
public:
  ~IdempotentRegionSequentializer() noexcept override;

  explicit IdempotentRegionSequentializer(rvsdg::Region & region);

  Sequentialization
  GetSequentialization() override;

  bool
  HasMoreSequentializations() const noexcept override;

  void
  ComputeNextSequentialization() override;
};

// FIXME: add documentation
class RegionTreeSequentializer
{
public:
  virtual ~RegionTreeSequentializer() noexcept;

  RegionTreeSequentializer();

  RegionTreeSequentializer(const RegionTreeSequentializer &) = delete;

  RegionTreeSequentializer(RegionTreeSequentializer &&) = delete;

  RegionTreeSequentializer &
  operator=(const RegionTreeSequentializer &) = delete;

  RegionTreeSequentializer &
  operator=(RegionTreeSequentializer &&) = delete;

  virtual void
  ComputeNextSequentializations() = 0;

  virtual RegionSequentializer &
  GetRegionSequentializer(rvsdg::Region & region) = 0;

  virtual bool
  HasMoreSequentializations() const noexcept = 0;
};

// FIXME: add documentation
class IdempotentRegionTreeSequentializer final : public RegionTreeSequentializer
{
public:
  ~IdempotentRegionTreeSequentializer() noexcept override;

  explicit IdempotentRegionTreeSequentializer(rvsdg::Region & region);

  void
  ComputeNextSequentializations() override;

  RegionSequentializer &
  GetRegionSequentializer(rvsdg::Region & region) override;

  bool
  HasMoreSequentializations() const noexcept override;

private:
  void
  InitializeSequentializers(rvsdg::Region & region);

  std::unordered_map<rvsdg::Region *, std::unique_ptr<IdempotentRegionSequentializer>>
      Sequentializers_;
};

// FIXME: add documentation
class ExhaustiveRegionTreeSequentializer final : public RegionTreeSequentializer
{
public:
  ~ExhaustiveRegionTreeSequentializer() noexcept override;

  explicit ExhaustiveRegionTreeSequentializer(rvsdg::Region & region);

  void
  ComputeNextSequentializations() override;

  RegionSequentializer &
  GetRegionSequentializer(rvsdg::Region & region) override;

  bool
  HasMoreSequentializations() const noexcept override;

private:
  void
  InitializeSequentializers(rvsdg::Region & region);

  std::unordered_map<rvsdg::Region *, std::unique_ptr<ExhaustiveRegionSequentializer>>
      Sequentializers_;
};

}

#endif // JLM_LLVM_BACKEND_REGIONSEQUENTIALIZER_HPP

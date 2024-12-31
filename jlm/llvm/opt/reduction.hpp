/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_REDUCTION_HPP
#define JLM_LLVM_OPT_REDUCTION_HPP

#include <jlm/llvm/opt/optimization.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::rvsdg
{
class Region;
class StructuralNode;
}

namespace jlm::llvm
{

/**
 * The node reduction transformation performs a series of peephole optimizations in the RVSDG. The
 * nodes in a region are visited top-down and reductions are performed until a fix-point is reached,
 * i.e., until no peephole optimization can be applied any longer to any node in a region.
 */
class NodeReduction final : public optimization
{
public:
  class Statistics;

  ~NodeReduction() noexcept override;

  NodeReduction();

  NodeReduction(const NodeReduction &) = delete;

  NodeReduction(NodeReduction &&) = delete;

  NodeReduction &
  operator=(const NodeReduction &) = delete;

  NodeReduction &
  operator=(NodeReduction &&) = delete;

  void
  run(RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  void
  run(RvsdgModule & rvsdgModule);

private:
  void
  ReduceNodesInRegion(rvsdg::Region & region);

  bool
  ReduceStructuralNode(rvsdg::StructuralNode & structuralNode);

  static bool
  ReduceGammaNode(rvsdg::StructuralNode & gammaNode);

  static bool
  ReduceSimpleNode(rvsdg::Node & simpleNode);

  static bool
  ReduceLoadNode(rvsdg::Node & simpleNode);

  static bool
  ReduceStoreNode(rvsdg::Node & simpleNode);

  static bool
  ReduceBinaryNode(rvsdg::Node & simpleNode);

  static std::optional<std::vector<rvsdg::output *>>
  NormalizeLoadNode(
      const LoadNonVolatileOperation & operation,
      const std::vector<rvsdg::output *> & operands);

  static std::optional<std::vector<rvsdg::output *>>
  NormalizeStoreNode(
      const StoreNonVolatileOperation & operation,
      const std::vector<rvsdg::output *> & operands);

  std::unique_ptr<Statistics> Statistics_;
};

/**
 * Represents the statistics gathered throughout the NodeReduction transformation.
 */
class NodeReduction::Statistics final : public util::Statistics
{
public:
  ~Statistics() noexcept override = default;

  explicit Statistics(const util::filepath & sourceFile)
      : util::Statistics(Id::ReduceNodes, sourceFile)
  {}

  void
  Start(const rvsdg::Graph & graph) noexcept;

  void
  End(const rvsdg::Graph & graph) noexcept;

  bool
  AddIteration(const rvsdg::Region & region, size_t numIterations);

  std::optional<size_t>
  GetNumIterations(const rvsdg::Region & region) const noexcept;

  static std::unique_ptr<Statistics>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }

private:
  std::unordered_map<const rvsdg::Region *, size_t> NumIterations_;
};

}

#endif

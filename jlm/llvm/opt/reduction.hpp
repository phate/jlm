/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_REDUCTION_HPP
#define JLM_LLVM_OPT_REDUCTION_HPP

#include <jlm/rvsdg/Transformation.hpp>
#include <jlm/util/Statistics.hpp>

#include <optional>

namespace jlm::rvsdg
{
class Graph;
class Node;
class Region;
class output;
class StructuralNode;
}

namespace jlm::llvm
{

class LoadNonVolatileOperation;
class MemoryStateMergeOperation;
class MemoryStateSplitOperation;
class StoreNonVolatileOperation;

/**
 * The node reduction transformation performs a series of peephole optimizations in the RVSDG. The
 * nodes in a region are visited top-down and reductions are performed until a fix-point is reached,
 * i.e., until no peephole optimization can be applied any longer to any node in a region.
 */
class NodeReduction final : public rvsdg::Transformation
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
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

private:
  void
  ReduceNodesInRegion(rvsdg::Region & region);

  /**
   * Reduces the structural node \p structuralNode.
   *
   * \note This method only returns true if the node itself could be reduced, but not if any of
   * the nodes in its subregions could be reduced.
   *
   * @param structuralNode The structural node that is supposed to be reduced.
   * @return True, if the structural node could be reduced, otherwise false.
   */
  [[nodiscard]] bool
  ReduceStructuralNode(rvsdg::StructuralNode & structuralNode);

  [[nodiscard]] static bool
  ReduceGammaNode(rvsdg::StructuralNode & gammaNode);

  [[nodiscard]] static bool
  ReduceSimpleNode(rvsdg::Node & simpleNode);

  [[nodiscard]] static bool
  ReduceLoadNode(rvsdg::Node & simpleNode);

  [[nodiscard]] static bool
  ReduceStoreNode(rvsdg::Node & simpleNode);

  [[nodiscard]] static bool
  ReduceMemoryStateMergeNode(rvsdg::Node & simpleNode);

  [[nodiscard]] static bool
  ReduceMemoryStateSplitNode(rvsdg::Node & simpleNode);

  [[nodiscard]] static bool
  ReduceBinaryNode(rvsdg::Node & simpleNode);

  static std::optional<std::vector<rvsdg::output *>>
  NormalizeLoadNode(
      const LoadNonVolatileOperation & operation,
      const std::vector<rvsdg::output *> & operands);

  static std::optional<std::vector<rvsdg::output *>>
  NormalizeStoreNode(
      const StoreNonVolatileOperation & operation,
      const std::vector<rvsdg::output *> & operands);

  static std::optional<std::vector<rvsdg::output *>>
  NormalizeMemoryStateMergeNode(
      const MemoryStateMergeOperation & operation,
      const std::vector<rvsdg::output *> & operands);

  static std::optional<std::vector<rvsdg::output *>>
  NormalizeMemoryStateSplitNode(
      const MemoryStateSplitOperation & operation,
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

  explicit Statistics(const util::FilePath & sourceFile)
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
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }

private:
  std::unordered_map<const rvsdg::Region *, size_t> NumIterations_;
};

}

#endif

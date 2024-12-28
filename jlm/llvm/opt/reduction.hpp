/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_REDUCTION_HPP
#define JLM_LLVM_OPT_REDUCTION_HPP

#include <jlm/llvm/opt/optimization.hpp>

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
  class Statistics;

public:
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
  static void
  ReduceNodesInRegion(rvsdg::Region & region);

  static bool
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
};

}

#endif

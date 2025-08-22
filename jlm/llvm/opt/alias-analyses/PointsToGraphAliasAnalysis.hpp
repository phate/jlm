/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_POINTSTOGRAPHALIASANALYSIS_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_POINTSTOGRAPHALIASANALYSIS_HPP

#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>

#include <string>

namespace jlm::llvm::aa
{

/**
 * Class for making alias analysis queries against a PointsToGraph
 */
class PointsToGraphAliasAnalysis : public AliasAnalysis
{
public:
  explicit PointsToGraphAliasAnalysis(const PointsToGraph & pointsToGraph);

  ~PointsToGraphAliasAnalysis() noexcept override;

  [[nodiscard]] std::string
  ToString() const override;

  AliasQueryResponse
  Query(const rvsdg::Output & p1, size_t s1, const rvsdg::Output & p2, size_t s2) override;

private:
  /**
   * Determines if there is a single valid target memory node for a given register node.
   * A target is only considered valid if it is large enough to hold the specified size.
   * If there are multiple valid targets, nullptr is returned.
   * @param node The register node to check the targets of
   * @param size The minimum size that the target memory node must be able to hold
   * @return a pointer to the single valid target memory node, or nullptr if there are zero or
   * multiple valid targets
   */
  [[nodiscard]] static const PointsToGraph::MemoryNode *
  TryGetSingleTarget(const PointsToGraph::RegisterNode & node, size_t size);

  /**
   * Determines the size of the memory region represented by the given memory node, if possible.
   * If the memory node represents multiple regions of the same size,
   * (e.g., an ALLOCA[i32]), the size of each represented region (e.g., 4) is returned.
   * @param node the MemoryNode representing an abstract memory location.
   * @return the size of the memory region, in bytes
   */
  [[nodiscard]] static std::optional<size_t>
  GetMemoryNodeSize(const PointsToGraph::MemoryNode & node);

  /**
   * Determines if the given abstract memory location represent exactly one region in memory,
   * such as imports and global variables.
   * As a counterexample, an ALLOCA[i32] can represent multiple 4-byte locations.
   * @param node the MemoryNode for the abstract memory location in question
   * @return true if node represents a single location
   */
  [[nodiscard]] static bool
  IsRepresentingSingleMemoryLocation(const PointsToGraph::MemoryNode & node);

  const PointsToGraph & PointsToGraph_;
};

} // namespace jlm::llvm::aa

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_POINTSTOGRAPHALIASANALYSIS_HPP

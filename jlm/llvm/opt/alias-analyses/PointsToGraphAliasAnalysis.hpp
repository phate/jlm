
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
  explicit PointsToGraphAliasAnalysis(PointsToGraph & pointsToGraph);

  ~PointsToGraphAliasAnalysis() override;

  [[nodiscard]] std::string
  ToString() const override;

  AliasQueryResponse
  Query(const rvsdg::Output & p1, size_t s1, const rvsdg::Output & p2, size_t s2) override;

private:
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

  PointsToGraph & PointsToGraph_;
};

} // namespace jlm::llvm::aa

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_POINTSTOGRAPHALIASANALYSIS_HPP

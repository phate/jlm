/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_LOADCHAINSEPARATION_HPP
#define JLM_LLVM_OPT_LOADCHAINSEPARATION_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class GammaNode;
class LambdaNode;
class ThetaNode;
}

namespace jlm::llvm
{

/**
 * Separates chains of memory region references from each other by rendering them independent in the
 * RVSDG through the insertion of a \ref MemoryStateJoinOperation node.
 *
 * The following example illustrates the transformation:
 *
 * v1 s2 = LoadNonVolatileOperation a1 s1
 * v2 s3 = LoadNonVolatileOperation a2 s2
 *
 * is transformed to:
 *
 * v1 s2 = LoadNonVolatileOperation a1 s1
 * v2 s3 = LoadNonVolatileOperation a2 s1
 * s4 = MemoryStateJoinOperation s2 s3
 */
class LoadChainSeparation final : public rvsdg::Transformation
{
  class Context;

  // FIXME: I really would like to rename this pass to ReferenceChainSeparation
public:
  ~LoadChainSeparation() noexcept override;

  LoadChainSeparation();

  LoadChainSeparation(const LoadChainSeparation &) = delete;

  LoadChainSeparation &
  operator=(const LoadChainSeparation &) = delete;

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
  void
  separateReferenceChainsInRegion(rvsdg::Region & region);

  void
  separateReferenceChainsInLambda(rvsdg::LambdaNode & lambdaNode);

  void
  separateRefenceChainsInTheta(rvsdg::ThetaNode & thetaNode);

  void
  separateRefenceChainsInGamma(rvsdg::GammaNode & gammaNode);

  /**
   * Separates the reference links of the mod/ref chain starting at memory state output \p
   * startOutput.
   *
   * @param startOutput The starting output of the mod/ref chain. Must be of type \ref
   * MemoryStateType.
   * @return True, if the separated mod/ref chains had a modifier link, otherwise False.
   */
  bool
  separateReferenceChains(rvsdg::Output & startOutput);

  /**
   * Represents a single link in a mod/ref chain
   */
  struct ModRefChainLink
  {
    /**
     * The type of the mod/ref chain link
     */
    enum class Type
    {
      /**
       * The link modifies the memory region.
       */
      Modification,

      /**
       * The link only references the memory region.
       */
      Reference,
    };

    /**
     * The memory state output associated with the node that references/modifies a memory region.
     */
    rvsdg::Output * output;
    Type type;
  };

  struct ModRefChain
  {
    void
    add(ModRefChainLink modRefChainLink)
    {
      links.push_back(std::move(modRefChainLink));
    }

    std::vector<ModRefChainLink> links{};
  };

  struct ModRefChainSummary
  {
    void
    add(ModRefChain modRefChain)
    {
      // We only care about chains that have at least two links
      if (modRefChain.links.size() >= 2)
      {
        modRefChains.push_back(std::move(modRefChain));
      }
    }

    std::vector<ModRefChain> modRefChains{};
  };

  /**
   * Recursively traces from output \p startOutput upwards to find all mod/ref chains
   * within a single region.
   *
   * @param startOutput The starting output for the tracing. Must be of type \ref MemoryStateType.
   * @param summary The tracing summary.
   *
   * @return True, if there is a ModRefChainLink::Type::Modification happening on any output above
   * \p startOutput in the region, otherwise false.
   */
  bool
  traceModRefChains(rvsdg::Output & startOutput, ModRefChainSummary & summary);

  /**
   * Extracts all reference subchains of mod/ref chain \p modRefChain. A valid reference subchain
   * is defined as followed:
   * 1. Must only contain mod/ref chain links of type \ref ModRefChainLink::Type::Reference
   * 2. Must have at least two links
   *
   * @param modRefChain The mod/ref chain from which to extract the reference subchains
   * @return A vector of reference subchains.
   */
  static std::vector<ModRefChain>
  extractReferenceSubchains(const ModRefChain & modRefChain);

  /**
   * Maps a memory state output of a node to the respective memory state input.
   *
   * @param output The output that is mapped.
   * @return A memory state input, if the output can be mapped.
   */
  static rvsdg::Input &
  mapMemoryStateOutputToInput(const rvsdg::Output & output);

  std::unique_ptr<Context> Context_{};
};

}

#endif

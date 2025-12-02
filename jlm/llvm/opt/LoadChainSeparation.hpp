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
  static void
  separateReferenceChainsInRegion(rvsdg::Region & region);

  static void
  separateReferenceChainsInLambda(rvsdg::LambdaNode & lambdaNode);

  static void
  separateRefenceChainsInTheta(rvsdg::ThetaNode & thetaNode);

  static void
  separateRefenceChainsInGamma(rvsdg::GammaNode & gammaNode);

  /**
   * Separates the reference links of the mod/ref chain starting at memory state output \p
   * startOutput.
   *
   * @param startOutput The starting output of the mod/ref chain. Must be of type \ref
   * MemoryStateType.
   * @param visitedOutputs The set of outputs that were already visited throughout the separation in
   * the region.
   */
  static void
  separateReferenceChains(
      rvsdg::Output & startOutput,
      util::HashSet<rvsdg::Output *> & visitedOutputs);

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
    std::vector<ModRefChainLink> links{};
  };

  /**
   * Recursively traces from output \p startOutput upwards to find all mod/ref chains
   * within a single region.
   *
   * @param startOutput The starting output for the tracing. Must be of type \ref MemoryStateType.
   * @param visitedOutputs The set of outputs that were already visited throughout the recursive
   * tracing.
   * @return A vector of mod/ref chains.
   */
  static std::vector<ModRefChain>
  traceModRefChains(rvsdg::Output & startOutput, util::HashSet<rvsdg::Output *> & visitedOutputs);

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
};

}

#endif

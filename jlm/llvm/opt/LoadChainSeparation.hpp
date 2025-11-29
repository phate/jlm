/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_LOADCHAINSEPARATION_HPP
#define JLM_LLVM_OPT_LOADCHAINSEPARATION_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class LambdaNode;
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
  separateModRefChainsInRegion(rvsdg::Region & region);

  /**
   * Separates the reference links of the mod/ref chain starting at memory state input \p
   * startInput.
   *
   * @param startInput The starting input of the mod/ref chain. Must be of type \ref
   * MemoryStateType.
   */
  static void
  separateModRefChains(rvsdg::Input & startInput);

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
     * The memory state input associated with the node that references/modifies a memory region.
     */
    rvsdg::Input * input;
    Type type;
  };

  struct ModRefChain
  {
    std::vector<ModRefChainLink> links{};
  };

  /**
   * Recursively traces from input \p startInput upwards to find all mod/ref chains
   * within a single region.
   *
   * @param startInput The starting input for the tracing. Must be of type \ref MemoryStateType.
   * @param visitedInputs The set of inputs that were already visited throughout the recursive
   * tracing.
   * @return A vector of mod/ref chains.
   */
  static std::vector<ModRefChain>
  traceModRefChains(rvsdg::Input & startInput, util::HashSet<rvsdg::Input *> & visitedInputs);

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
   * Maps a memory state input of a node to the respective memory state output.
   *
   * @param input The input that is mapped.
   * @return A memory state output, if the input can be mapped.
   */
  static rvsdg::Output &
  mapMemoryStateInputToOutput(const rvsdg::Input & input);
};

}

#endif

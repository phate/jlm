/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_LOADCHAINSEPARATION_HPP
#define JLM_LLVM_OPT_LOADCHAINSEPARATION_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::llvm
{

/**
 * Separates chains of \ref LoadOperation nodes from each other by rendering them independent in the
 * RVSDG through the insertion of a \ref MemoryStateJoinOperation node.
 *
 * The following example illustrate the transformation:
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
  handleRegion(rvsdg::Region & region);

  /**
   * Takes the memory state output of a \ref LoadOperation node and separates all the load nodes
   * that are connected to the respective memory state edge above the graph by inserting a \ref
   * MemoryStateJoinOperation node.
   */
  static void
  separateLoadChain(rvsdg::Output & memoryStateOutput);

  /**
   * Traces the memory state output of a \ref LoadOperation upwards through the load node chain,
   * collecting the memory state outputs of each \ref LoadOperation node along the way, and
   * returning the final output that is not owned by a \ref LoadOperation node.
   *
   * @param output The memory state output of a \ref LoadOperation node.
   * @param [out] joinOperands A vector of all the memory state outputs encountered while tracing up
   * the load chain.
   * @return The final output that is not owned by a \ref LoadOperation node.
   */
  static rvsdg::Output &
  traceLoadNodeMemoryState(rvsdg::Output & output, std::vector<rvsdg::Output *> & joinOperands);

  /**
   * Finds all memory state outputs of a \ref LoadOperation node that does not have a \ref
   * LoadOperation node as owner of its users.
   *
   * @param region The region in which to find the memory state outputs.
   * @return A set of memory state outputs.
   */
  static util::HashSet<rvsdg::Output *>
  findLoadChainBottoms(rvsdg::Region & region);

  static void
  findLoadChainEnds(rvsdg::Region & region, util::HashSet<rvsdg::Output *> & loadChainEnds);

  /**
   * @return True, if the origin of \p input is a \ref LoadOperation node, otherwise false.
   */
  static bool
  hasLoadNodeAsOperandOwner(const rvsdg::Input & input);

  /**
   * @return True, if \p output has a \ref LoadOperation node as one of its users' owners, otherwise
   * false.
   */
  static bool
  hasLoadNodeAsUserOwner(const rvsdg::Output & output);
};

}

#endif

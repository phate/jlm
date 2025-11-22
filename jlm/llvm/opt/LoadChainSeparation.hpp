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
  // FIXME: I really would like to rename this pass to ModRefChainSeparation
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
  separateModRefChainsInRegion(rvsdg::Region & region);

  // FIXME: documentation
  void
  separateModRefChains(rvsdg::Input & input);

  // FIXME: documentation
  enum class ModRefChainLinkType
  {
    Modification,
    Reference,
  };

  struct ModRefChainLink
  {
    rvsdg::Input * input;
    ModRefChainLinkType modRefType;
  };

  struct ModRefChain
  {
    std::vector<ModRefChainLink> links{};
  };

  // FIXME: documentation
  std::vector<ModRefChain>
  traceModRefChains(rvsdg::Input & startInput);

  // FIXME: documentation
  std::vector<std::pair<size_t, size_t>>
  computeReferenceSubchains(const ModRefChain & modRefChain);

  // FIXME: documentation
  rvsdg::Output &
  mapMemoryStateInputToOutput(const rvsdg::Input & input);
};

}

#endif

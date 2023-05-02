/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_OPTIMIZATION_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_OPTIMIZATION_HPP

#include <jlm/llvm/opt/optimization.hpp>

namespace jlm::aa {

/** \brief Steensgaard alias analysis with agnostic memory state encoding
 *
 * @see Steensgaard
 * @see AgnosticMemoryNodeProvider
 */
class SteensgaardAgnostic final : public optimization {
public:
  ~SteensgaardAgnostic() noexcept override;

  void
  run(
    RvsdgModule & rvsdgModule,
    StatisticsCollector & statisticsCollector) override;
};

/** \brief Steensgaard alias analysis with region-aware memory state encoding
 *
 * @see Steensgaard
 * @see RegionAwareMemoryNodeProvider
 */
class SteensgaardRegionAware final : public optimization {
public:
  ~SteensgaardRegionAware() noexcept override;

  void
  run(
    RvsdgModule & rvsdgModule,
    StatisticsCollector & statisticsCollector) override;
};

}

#endif

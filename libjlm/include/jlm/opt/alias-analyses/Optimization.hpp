/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_OPTIMIZATION_HPP
#define JLM_OPT_ALIAS_ANALYSES_OPTIMIZATION_HPP

#include <jlm/opt/optimization.hpp>

namespace jlm::aa {

/** \brief Steensgaard alias analysis with basic memory state encoding
 *
 * @see Steensgaard
 * @see BasicMemoryNodeProvider
 */
class SteensgaardBasic final : public optimization {
public:
  ~SteensgaardBasic() noexcept override;

  void
  run(
    RvsdgModule & rvsdgModule,
    const StatisticsDescriptor & statisticsDescriptor) override;
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
    const StatisticsDescriptor & statisticsDescriptor) override;
};

}

#endif

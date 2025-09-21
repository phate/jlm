/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_GAMMACONVERSION_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_GAMMACONVERSION_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::hls
{

/**
 * Converts every rvsdg::GammaNode in \p rvsdgModule to its respective HLS equivalent.
 *
 * @param rvsdgModule The RVSDG module the transformation is performed on.
 */
class GammaNodeConversion final : public rvsdg::Transformation
{
public:
  ~GammaNodeConversion() noexcept override;

  GammaNodeConversion();

  GammaNodeConversion(const GammaNodeConversion &) = delete;

  GammaNodeConversion &
  operator=(const GammaNodeConversion &) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
  {
    GammaNodeConversion gammaNodeConversion;
    gammaNodeConversion.Run(rvsdgModule, statisticsCollector);
  }
};

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_GAMMACONVERSION_HPP

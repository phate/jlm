/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/opt/alias-analyses/BasicEncoder.hpp>
#include <jlm/opt/alias-analyses/Optimization.hpp>
#include <jlm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/opt/alias-analyses/Steensgaard.hpp>

namespace jlm::aa {

SteensgaardBasic::~SteensgaardBasic() noexcept
= default;

void
SteensgaardBasic::run(
  RvsdgModule & rvsdgModule,
  const StatisticsDescriptor & statisticsDescriptor)
{
  Steensgaard steensgaard;
  auto pointsToGraph = steensgaard.Analyze(rvsdgModule, statisticsDescriptor);

  BasicEncoder encoder(*pointsToGraph);
  encoder.Encode(rvsdgModule, statisticsDescriptor);
}

}

/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/AgnosticMemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryStateEncoder.hpp>
#include <jlm/llvm/opt/alias-analyses/Optimization.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareMemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>

namespace jlm::llvm::aa
{

SteensgaardAgnostic::~SteensgaardAgnostic() noexcept
= default;

void
SteensgaardAgnostic::run(
  RvsdgModule & rvsdgModule,
  jlm::util::StatisticsCollector & statisticsCollector)
{
  Steensgaard steensgaard;
  auto pointsToGraph = steensgaard.Analyze(rvsdgModule, statisticsCollector);

  auto provisioning = AgnosticMemoryNodeProvider::Create(rvsdgModule, *pointsToGraph, statisticsCollector);

  MemoryStateEncoder encoder;
  encoder.Encode(rvsdgModule, *provisioning, statisticsCollector);
}

SteensgaardRegionAware::~SteensgaardRegionAware() noexcept
= default;

void
SteensgaardRegionAware::run(
  RvsdgModule & rvsdgModule,
  util::StatisticsCollector & statisticsCollector)
{
  Steensgaard steensgaard;
  auto pointsToGraph = steensgaard.Analyze(rvsdgModule, statisticsCollector);

  auto provisioning = RegionAwareMemoryNodeProvider::Create(rvsdgModule, *pointsToGraph, statisticsCollector);

  MemoryStateEncoder encoder;
  encoder.Encode(rvsdgModule, *provisioning, statisticsCollector);
}

}
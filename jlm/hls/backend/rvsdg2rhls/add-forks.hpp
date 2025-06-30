/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_ADD_FORKS_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_ADD_FORKS_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::hls
{

class ForkInsertion final : public rvsdg::Transformation
{
public:
  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
  static void
  AddForksToRegion(rvsdg::Region & region);

  static void
  AddForkToOutput(rvsdg::Output & output);

  [[nodiscard]] static bool
  IsConstantFork(const rvsdg::Output & output);
};

/**
 * Adds a fork for every output that has multiple consumers (node inputs). The original output is
 * connected to the fork's input and each consumer is connected to one of the fork's outputs.
 *
 * /param region The region for which to insert forks.
 */
void
add_forks(rvsdg::Region * region);

/**
 * Adds a fork for every output that has multiple consumers (node inputs). The original output is
 * connected to the fork's input and each consumer is connected to one of the fork's outputs.
 *
 * /param rvsdgModule The RVSDG module for which to insert forks.
 */
void
add_forks(llvm::RvsdgModule & rvsdgModule);

}
#endif // JLM_HLS_BACKEND_RVSDG2RHLS_ADD_FORKS_HPP

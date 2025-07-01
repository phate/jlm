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

/**
 * Adds fork nodes for every output that has multiple users to ensure that each output has at most
 * a single user. The original output is connected to the fork's input and each user is connected to
 * one of the fork's outputs.
 */
class ForkInsertion final : public rvsdg::Transformation
{
public:
  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector);

private:
  static void
  AddForksToRegion(rvsdg::Region & region);

  static void
  AddForkToOutput(rvsdg::Output & output);

  [[nodiscard]] static bool
  IsConstantFork(const rvsdg::Output & output);
};

}
#endif // JLM_HLS_BACKEND_RVSDG2RHLS_ADD_FORKS_HPP

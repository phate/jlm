/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_CNE_HPP
#define JLM_LLVM_OPT_CNE_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::llvm
{

/**
 * \brief Common Node Elimination
 * Discovers simple nodes, region arguments and structural node outputs that are guaranteed to
 * always produce the same value, and redirects all their users to the same output.
 * This renders common nodes and common structural arguments / results dead.
 */
class CommonNodeElimination final : public rvsdg::Transformation
{
public:
  class Context;
  class Statistics;

  ~CommonNodeElimination() noexcept override;

  CommonNodeElimination()
      : Transformation("CommonNodeElimination")
  {}

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;
};

}

#endif

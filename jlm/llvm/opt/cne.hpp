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
 */
class CommonNodeElimination final : public rvsdg::Transformation
{
public:
  class Statistics;

  ~CommonNodeElimination() noexcept override;

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;
};

}

#endif

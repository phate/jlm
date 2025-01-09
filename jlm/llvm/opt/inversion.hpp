/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_INVERSION_HPP
#define JLM_LLVM_OPT_INVERSION_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::llvm
{

/**
 * \brief Theta-Gamma Inversion
 */
class tginversion final : public rvsdg::Transformation
{
public:
  virtual ~tginversion();

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;
};

}

#endif

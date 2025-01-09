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
class cne final : public rvsdg::Transformation
{
public:
  virtual ~cne();

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;
};

}

#endif

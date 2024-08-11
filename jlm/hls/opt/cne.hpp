/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_OPT_CNE_HPP
#define JLM_HLS_OPT_CNE_HPP

#include <jlm/llvm/opt/optimization.hpp>

namespace jlm::llvm
{
class RvsdgModule;
}

namespace jlm::hls
{

/**
 * \brief Common Node Elimination
 */
class cne final : public llvm::optimization
{
public:
  virtual ~cne();

  virtual void
  run(llvm::RvsdgModule & module, jlm::util::StatisticsCollector & statisticsCollector) override;
};

}

#endif

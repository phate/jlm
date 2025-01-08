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

// FIXME
// The cne optimization should be generalized such that it can be used for both the LLVM and HLS
// backend.

/**
 * \brief Common Node Elimination
 * This is mainly a copy of the CNE optimization in the LLVM backend with the addition of support
 * for the hls::loop_op.
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

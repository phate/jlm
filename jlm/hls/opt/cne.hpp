/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_OPT_CNE_HPP
#define JLM_HLS_OPT_CNE_HPP

#include <jlm/rvsdg/Transformation.hpp>

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
class cne final : public rvsdg::Transformation
{
public:
  ~cne() noexcept override;

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;
};

}

#endif

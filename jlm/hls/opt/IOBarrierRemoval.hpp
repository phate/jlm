/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_OPT_IOBARRIERREMOVAL_HPP
#define JLM_HLS_OPT_IOBARRIERREMOVAL_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class Region;
}

namespace jlm::hls
{

/**
 * \brief Removes all IOBarrier nodes from the RVSDG.
 *
 * In HLS, we can safely assume that we will not encounter any undefined behavior. However, this
 * means also that we can relax the sequentialization restrictions on certain operations, such as
 * division or modulo operations, with respect to each other. Ultimately, it means that we can
 * remove IOBarrier nodes from the RVSDG when performing HLS.
 *
 * @see IOBarrierOperation
 */
class IOBarrierRemoval final : public rvsdg::Transformation
{
public:
  ~IOBarrierRemoval() noexcept override;

  IOBarrierRemoval()
      : Transformation("IOBarrierRemoval")
  {}

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
  static void
  RemoveIOBarrierFromRegion(rvsdg::Region & region);
};

}

#endif // JLM_HLS_OPT_IOBARRIERREMOVAL_HPP

/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_MEM_QUEUE_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_MEM_QUEUE_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::hls
{

class AddressQueueInsertion final : public rvsdg::Transformation
{
public:
  ~AddressQueueInsertion() noexcept override;

  AddressQueueInsertion();

  AddressQueueInsertion(const AddressQueueInsertion &) = delete;

  AddressQueueInsertion &
  operator=(const AddressQueueInsertion &) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
  {
    AddressQueueInsertion addressQueueInsertion;
    addressQueueInsertion.Run(rvsdgModule, statisticsCollector);
  }
};

}

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_MEM_QUEUE_HPP

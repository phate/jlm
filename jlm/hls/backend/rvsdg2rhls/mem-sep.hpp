//
// Created by david on 7/2/21.
//

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_MEM_SEP_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_MEM_SEP_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class LambdaNode;
}

namespace jlm::hls
{

class MemoryStateSeparation final : public rvsdg::Transformation
{
public:
  ~MemoryStateSeparation() noexcept override;

  MemoryStateSeparation();

  MemoryStateSeparation(const MemoryStateSeparation &) = delete;

  MemoryStateSeparation(MemoryStateSeparation &&) = delete;

  MemoryStateSeparation &
  operator=(const MemoryStateSeparation &) = delete;

  MemoryStateSeparation &
  operator=(MemoryStateSeparation &&) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
  {
    MemoryStateSeparation memoryStateSeparation;
    memoryStateSeparation.Run(rvsdgModule, statisticsCollector);
  }

private:
  static void
  separateMemoryStates(const rvsdg::LambdaNode & lambdaNode);
};

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_MEM_SEP_HPP

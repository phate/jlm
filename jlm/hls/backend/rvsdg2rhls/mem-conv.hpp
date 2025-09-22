/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_MEM_CONV_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_MEM_CONV_HPP

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::hls
{

typedef std::vector<std::tuple<
    std::vector<jlm::rvsdg::SimpleNode *>,
    std::vector<jlm::rvsdg::SimpleNode *>,
    std::vector<jlm::rvsdg::SimpleNode *>>>
    port_load_store_decouple;

/**
 * Traces all pointer arguments of a lambda node and finds all memory operations.
 * Pointers read from memory is not traced, i.e., the output of load operations is not traced.
 * @param lambda The lambda node for which to trace all pointer arguments
 * @param portNodes A vector where each element contains all memory operations traced from a pointer
 */
void
TracePointerArguments(const rvsdg::LambdaNode * lambda, port_load_store_decouple & portNodes);

jlm::rvsdg::SimpleNode *
find_decouple_response(
    const jlm::rvsdg::LambdaNode * lambda,
    const jlm::llvm::IntegerConstantOperation * request_constant);

class MemoryConverter final : public rvsdg::Transformation
{
public:
  ~MemoryConverter() noexcept override;

  MemoryConverter();

  MemoryConverter(const MemoryConverter &) = delete;

  MemoryConverter &
  operator=(const MemoryConverter &) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
  {
    MemoryConverter memoryConverter;
    memoryConverter.Run(rvsdgModule, statisticsCollector);
  }
};

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_MEM_CONV_HPP

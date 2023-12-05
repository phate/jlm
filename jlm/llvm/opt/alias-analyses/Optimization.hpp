/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_OPTIMIZATION_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_OPTIMIZATION_HPP

#include "AliasAnalysis.hpp"
#include <jlm/llvm/opt/optimization.hpp>

#include <type_traits>

namespace jlm::llvm::aa
{

/** \brief Applies alias analysis and memory state encoding
 * Uses the information collected during alias analysis to
 *
 * The type of alias analysis is specified by the template parameter.
 * The second template parameters specifies wether or not to use the
 * RegionAwareMemoryNodeProvider or the AgnosticMemoryNodeProvider
 *
 * @tparam AliasAnalysisPass the subclass of AliasAnalysis to use
 * @tparam regionAware if true, the RegionAwareMemoryNodeProvider is used
 *
 * @see Steensgaard
 * @see Andersen
 * @see AgnosticMemoryNodeProvider
 * @see RegionAwareMemoryNodeProvider
 */
template<typename AliasAnalysisPass, bool regionAware>
class MemoryStateEncodingPass final : public optimization
{
  static_assert(std::is_base_of_v<AliasAnalysis, AliasAnalysisPass>);

public:
  ~MemoryStateEncodingPass() noexcept override;

  void
  run(RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;
};

}

#endif

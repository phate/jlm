/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_FRONTEND_INTERPROCEDURALGRAPHCONVERSION_HPP
#define JLM_LLVM_FRONTEND_INTERPROCEDURALGRAPHCONVERSION_HPP

#include <memory>


namespace jlm::util
{
class StatisticsCollector;
}

namespace jlm::llvm
{

class ipgraph_module;
class RvsdgModule;

std::unique_ptr<RvsdgModule>
ConvertInterProceduralGraphModule(
  const ipgraph_module & im,
  jlm::util::StatisticsCollector & statisticsCollector);

}

#endif

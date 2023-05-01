/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_FRONTEND_LLVM_INTERPROCEDURALGRAPHCONVERSION_HPP
#define JLM_FRONTEND_LLVM_INTERPROCEDURALGRAPHCONVERSION_HPP

#include <memory>

namespace jlm {

class ipgraph_module;
class RvsdgModule;
class StatisticsCollector;

std::unique_ptr<RvsdgModule>
ConvertInterProceduralGraphModule(
  const ipgraph_module & im,
  StatisticsCollector & statisticsCollector);

}

#endif

/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_BACKEND_RVSDG2JLM_RVSDG2JLM_HPP
#define JLM_LLVM_BACKEND_RVSDG2JLM_RVSDG2JLM_HPP

#include <memory>

namespace jive {

class graph;

}

namespace jlm {

class ipgraph_module;
class RvsdgModule;
class StatisticsCollector;

namespace rvsdg2jlm {

std::unique_ptr<ipgraph_module>
rvsdg2jlm(
  const RvsdgModule & rm,
  StatisticsCollector & statisticsCollector);

}}

#endif

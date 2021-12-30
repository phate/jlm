/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_FRONTEND_LLVM_JLM2RVSDG_MODULE_H
#define JLM_FRONTEND_LLVM_JLM2RVSDG_MODULE_H

#include <memory>

namespace jive {
	class graph;
}

namespace jlm {

class ipgraph_module;
class rvsdg_module;
class StatisticsDescriptor;

std::unique_ptr<rvsdg_module>
construct_rvsdg(const ipgraph_module & im, const StatisticsDescriptor & sd);

}

#endif

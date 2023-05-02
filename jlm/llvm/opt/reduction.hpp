/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_REDUCTION_HPP
#define JLM_LLVM_OPT_REDUCTION_HPP

#include <jlm/llvm/opt/optimization.hpp>

namespace jlm {

class rvsdg;

/**
* \brief Node Reduction Optimization
*/
class nodereduction final : public optimization {
public:
	virtual
	~nodereduction();

	virtual void
	run(
    RvsdgModule & module,
    StatisticsCollector & statisticsCollector) override;
};

}

#endif

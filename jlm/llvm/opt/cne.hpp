/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_CNE_HPP
#define JLM_LLVM_OPT_CNE_HPP

#include <jlm/llvm/opt/optimization.hpp>

namespace jlm {

class RvsdgModule;

/**
* \brief Common Node Elimination
*/
class cne final : public optimization {
public:
	virtual
	~cne();

	virtual void
	run(
    RvsdgModule & module,
    StatisticsCollector & statisticsCollector) override;
};

}

#endif

/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_OPTIMIZATION_HPP
#define JLM_OPT_ALIAS_ANALYSES_OPTIMIZATION_HPP

#include <jlm/opt/optimization.hpp>

namespace jlm {
namespace aa {

class SteensgaardBasic final : public optimization {
public:
	virtual
	~SteensgaardBasic() override;

	virtual void
	run(rvsdg_module & module, const StatisticsDescriptor & sd) override;
};

}}

#endif

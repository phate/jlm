/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/opt/alias-analyses/BasicEncoder.hpp>
#include <jlm/opt/alias-analyses/Optimization.hpp>
#include <jlm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/opt/alias-analyses/Steensgaard.hpp>

namespace jlm {
namespace aa {

SteensgaardBasic::~SteensgaardBasic() = default;

void
SteensgaardBasic::run(
	rvsdg_module & module,
	const StatisticsDescriptor & sd)
{
	Steensgaard steensgaard;
	auto ptg = steensgaard.Analyze(module, sd);

	BasicEncoder encoder(*ptg);
	encoder.Encode(module);
}

}}

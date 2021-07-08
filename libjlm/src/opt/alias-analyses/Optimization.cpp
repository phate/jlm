/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/opt/alias-analyses/encoders.hpp>
#include <jlm/opt/alias-analyses/Optimization.hpp>
#include <jlm/opt/alias-analyses/pointsto-graph.hpp>
#include <jlm/opt/alias-analyses/steensgaard.hpp>

namespace jlm {
namespace aa {

SteensgaardBasic::~SteensgaardBasic()
{}

void
SteensgaardBasic::run(
	rvsdg_module & module,
	const stats_descriptor & sd)
{
	Steensgaard steensgaard;
	auto ptg = steensgaard.Analyze(module);

//	std::cout << jlm::aa::ptg::to_dot(*ptg) << std::flush;

	BasicEncoder encoder(*ptg);
	encoder.Encode(module);
}

}}

/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/rvsdg.hpp>

#include <jlm/opt/cne.hpp>
#include <jlm/opt/dne.hpp>
#include <jlm/opt/inlining.hpp>
#include <jlm/opt/invariance.hpp>
#include <jlm/opt/inversion.hpp>
#include <jlm/opt/optimization.hpp>
#include <jlm/opt/pull.hpp>
#include <jlm/opt/push.hpp>
#include <jlm/opt/reduction.hpp>
#include <jlm/opt/unroll.hpp>

#include <jlm/util/stats.hpp>
#include <jlm/util/time.hpp>

#include <unordered_map>

namespace jlm {

void
optimize(rvsdg_module & rm, const optimization & opt)
{
	static std::unordered_map<optimization, void(*)(rvsdg_module&)> map({
	  {optimization::cne, jlm::cne }
	, {optimization::dne, jlm::dne }
	, {optimization::iln, jlm::inlining }
	, {optimization::inv, jlm::invariance }
	, {optimization::pll, jlm::pull }
	, {optimization::psh, jlm::push }
	, {optimization::ivt, jlm::invert }
	, {optimization::url, [](rvsdg_module & rm){ jlm::unroll(rm, 4); }}
	, {optimization::red, jlm::reduce }
	});


	JLM_DEBUG_ASSERT(map.find(opt) != map.end());
	map[opt](rm);
}

void
optimize(
	rvsdg_module & rm,
	const std::vector<optimization> & opts,
	const stats_descriptor & sd)
{
	jlm::timer timer;
	size_t nnodes_before = 0;
	if (sd.print_rvsdg_optimization) {
		nnodes_before = jive::nnodes(rm.graph()->root());
		timer.start();
	}

	for (const auto & opt : opts)
		optimize(rm, opt);

	if (sd.print_rvsdg_optimization) {
		timer.stop();
		size_t nnodes_after = jive::nnodes(rm.graph()->root());
		fprintf(sd.file().fd(),
			"RVSDGOPTIMIZATION %s %zu %zu %zu\n", rm.source_filename().to_str().c_str(),
				nnodes_before, nnodes_after, timer.ns());
	}
}

}

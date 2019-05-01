/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

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

#include <unordered_map>

namespace jlm {

void
optimize(jive::graph & graph, const optimization & opt)
{
	static std::unordered_map<optimization, void(*)(jive::graph&)> map({
	  {optimization::cne, [](jive::graph & graph){ jlm::cne(graph); }}
	, {optimization::dne, [](jive::graph & graph){ jlm::dne(graph); }}
	, {optimization::iln, [](jive::graph & graph){ jlm::inlining(graph); }}
	, {optimization::inv, [](jive::graph & graph){ jlm::invariance(graph); }}
	, {optimization::pll, [](jive::graph & graph){ jlm::pull(graph); }}
	, {optimization::psh, [](jive::graph & graph){ jlm::push(graph); }}
	, {optimization::ivt, [](jive::graph & graph){ jlm::invert(graph); }}
	, {optimization::url, [](jive::graph & graph){ jlm::unroll(graph, 4); }}
	, {optimization::red, [](jive::graph & graph){ jlm::reduce(graph); }}
	});


	JLM_DEBUG_ASSERT(map.find(opt) != map.end());
	map[opt](graph);
}

}

/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/hls/hls.hpp>
#include <jive/rvsdg/traverser.hpp>
#include "jlm/backend/hls/rvsdg2rhls/add-sinks.hpp"

void
jlm::hls::add_sinks(jive::region *region) {
	for (size_t i = 0; i < region->narguments(); ++i) {
		auto arg = region->argument(i);
		if (!arg->nusers()) {
			hls::sink_op::create(*arg);
		}
	}
	for (auto &node : jive::topdown_traverser(region)) {
		if (auto structnode = dynamic_cast<jive::structural_node *>(node)) {
			for (size_t n = 0; n < structnode->nsubregions(); n++) {
				add_sinks(structnode->subregion(n));
			}
		}

		for (size_t i = 0; i < node->noutputs(); ++i) {
			auto out = node->output(i);
			if (!out->nusers()) {
				hls::sink_op::create(*out);
			}
		}
	}
}

void
jlm::hls::add_sinks(jlm::RvsdgModule &rm) {
	auto &graph = rm.Rvsdg();
	auto root = graph.root();
	add_sinks(root);
}

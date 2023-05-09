/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/check-rhls.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/traverser.hpp>

void
jlm::hls::check_rhls(jive::region *sr) {
	for (auto &node : jive::topdown_traverser(sr)) {
		if (dynamic_cast<jive::structural_node *>(node)) {
			if (auto ln = dynamic_cast<hls::loop_node *>(node)) {
				check_rhls(ln->subregion());
			} else {
				throw jlm::util::error("There should be only simple nodes and loop nodes");
			}
		}
		for (size_t i = 0; i < node->noutputs(); i++) {
			if (node->output(i)->nusers() == 0) {
				throw jlm::util::error("Output has no users");
			} else if (node->output(i)->nusers() > 1) {
				throw jlm::util::error("Output has more than one user");
			}
		}
		if (is_constant(node)) {
			if (node->noutputs() != 1) {
				throw jlm::util::error("Constant should have one output");
			}
			auto user_in = dynamic_cast<jive::node_input *>(*node->output(0)->begin());
			if (!user_in || !jive::is<hls::trigger_op>(user_in->node())) {
				throw jlm::util::error("Constant has to be gated by a trigger");
			}
		}
	}
}

void
jlm::hls::check_rhls(jlm::RvsdgModule &rm) {
	auto &graph = rm.Rvsdg();
	auto root = graph.root();
	if (root->nodes.size() != 1) {
		throw jlm::util::error("Root should have only one node now");
	}
	auto ln = dynamic_cast<const jlm::lambda::node *>(root->nodes.begin().ptr());
	if (!ln) {
		throw jlm::util::error("Node needs to be a lambda");
	}
	check_rhls(ln->subregion());
}

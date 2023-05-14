/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/add-buffers.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/rvsdg/traverser.hpp>

void
jlm::hls::add_buffers(jlm::rvsdg::region *region, bool pass_through) {
	for (auto &node : jlm::rvsdg::topdown_traverser(region)) {
		if (auto structnode = dynamic_cast<jlm::rvsdg::structural_node *>(node)) {
			for (size_t n = 0; n < structnode->nsubregions(); n++) {
				add_buffers(structnode->subregion(n), pass_through);
			}
		} else if (dynamic_cast<jlm::rvsdg::simple_node *>(node)) {
			if (jlm::rvsdg::is<hls::buffer_op>(node)) {
				continue;
			} else if (is_constant(node)) {
				continue;
			} else if (jlm::rvsdg::is<hls::print_op>(node)) {
				continue;
			}
			for (size_t i = 0; i < node->noutputs(); ++i) {
				auto out = node->output(i);
				JLM_ASSERT(out->nusers() == 1);
				if (auto ni = dynamic_cast<jlm::rvsdg::node_input *>(*out->begin())) {
					auto buf = dynamic_cast<const hls::buffer_op*>(&ni->node()->operation());
					if (buf && (buf->pass_through||!pass_through)) {
						continue;
					}
				}
				std::vector<jlm::rvsdg::input *> old_users(out->begin(), out->end());
				jlm::rvsdg::output* new_out;
				if(pass_through){
					new_out = buffer_op::create(*out, 10, true)[0];
				} else{
					new_out = buffer_op::create(*out, 1, false)[0];
				}
				for (auto user: old_users) {
					user->divert_to(new_out);
				}
			}
		}
	}
}

void
jlm::hls::add_buffers(llvm::RvsdgModule &rm, bool pass_through) {
	auto &graph = rm.Rvsdg();
	auto root = graph.root();
	add_buffers(root, pass_through);
}

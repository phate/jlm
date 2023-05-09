/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_RHLS_DNE_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_RHLS_DNE_HPP

#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm {
	namespace hls {
		bool
		remove_unused_loop_outputs(hls::loop_node *ln);

		bool
		remove_unused_loop_inputs(hls::loop_node *ln);


		bool
		dne(jlm::rvsdg::region *sr);

		void
		dne(jlm::RvsdgModule &rm);
	}
}
#endif //JLM_HLS_BACKEND_RVSDG2RHLS_RHLS_DNE_HPP

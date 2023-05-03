/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/gamma.hpp>

namespace jlm {
	namespace hls {

		bool
		is_passthrough(const jive::argument *arg);

		bool
		is_passthrough(const jive::result *res);

		jlm::lambda::node *
		remove_lambda_passthrough(jlm::lambda::node *ln);

		void
		remove_region_passthrough(const jive::argument *arg);

		void
		remove_gamma_passthrough(jive::gamma_node *gn);

		void
		remove_unused_state(jlm::RvsdgModule &rm);

		void
		remove_unused_state(jive::region *region, bool can_remove_arguments = true);
	}
}
#endif //JLM_BACKEND_HLS_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP

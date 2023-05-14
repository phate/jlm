/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/gamma.hpp>

namespace jlm {
	namespace hls {

		bool
		is_passthrough(const jlm::rvsdg::argument *arg);

		bool
		is_passthrough(const jlm::rvsdg::result *res);

		llvm::lambda::node *
		remove_lambda_passthrough(llvm::lambda::node *ln);

		void
		remove_region_passthrough(const jlm::rvsdg::argument *arg);

		void
		remove_gamma_passthrough(jlm::rvsdg::gamma_node *gn);

		void
		remove_unused_state(llvm::RvsdgModule &rm);

		void
		remove_unused_state(jlm::rvsdg::region *region, bool can_remove_arguments = true);
	}
}
#endif //JLM_HLS_BACKEND_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP

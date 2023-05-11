/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_ADD_TRIGGERS_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_ADD_TRIGGERS_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jlm {
	namespace hls {

		rvsdg::output *
		get_trigger(rvsdg::region *region);

		jlm::lambda::node *
		add_lambda_argument(jlm::lambda::node *ln, const rvsdg::type *type);

		void
		add_triggers(rvsdg::region *region);

		void
		add_triggers(jlm::RvsdgModule &rm);
	}
}
#endif //JLM_HLS_BACKEND_RVSDG2RHLS_ADD_TRIGGERS_HPP

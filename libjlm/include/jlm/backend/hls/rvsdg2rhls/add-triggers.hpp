/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LIBJLM_SRC_BACKEND_HLS_RVSDG2RHLS_ADD_TRIGGERS_HPP
#define JLM_LIBJLM_SRC_BACKEND_HLS_RVSDG2RHLS_ADD_TRIGGERS_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jlm {
	namespace hls {

		jive::output *
		get_trigger(jive::region *region);

		jlm::lambda::node *
		add_lambda_argument(jlm::lambda::node *ln, const jive::type *type);

		void
		add_triggers(jive::region *region);

		void
		add_triggers(jlm::RvsdgModule &rm);
	}
}
#endif //JLM_LIBJLM_SRC_BACKEND_HLS_RVSDG2RHLS_ADD_TRIGGERS_HPP

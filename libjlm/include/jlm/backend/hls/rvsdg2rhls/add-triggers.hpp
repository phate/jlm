//
// Created by david on 7/2/21.
//

#ifndef JLM_LIBJLM_SRC_BACKEND_HLS_RVSDG2RHLS_ADD_TRIGGERS_HPP
#define JLM_LIBJLM_SRC_BACKEND_HLS_RVSDG2RHLS_ADD_TRIGGERS_HPP

#include <jive/rvsdg/region.hpp>
#include <jlm/ir/operators/lambda.hpp>
#include <jlm/ir/RvsdgModule.hpp>

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

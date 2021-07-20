//
// Created by david on 7/2/21.
//

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_THETA_CONV_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_THETA_CONV_HPP

#include <jive/rvsdg/theta.hpp>
#include <jlm/ir/RvsdgModule.hpp>

namespace jlm{
	namespace hls{
		void theta_conv(jive::theta_node *theta);

		void
		theta_conv(jive::region *region);

		void
		theta_conv(jlm::RvsdgModule &rm);
	}
}
#endif //JLM_BACKEND_HLS_RVSDG2RHLS_THETA_CONV_HPP

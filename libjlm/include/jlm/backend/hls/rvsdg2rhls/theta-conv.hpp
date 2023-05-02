/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_THETA_CONV_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_THETA_CONV_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/theta.hpp>

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

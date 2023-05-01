/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_GAMMA_CONV_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_GAMMA_CONV_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>

namespace jlm{
	namespace hls{

		bool gamma_can_be_spec(jive::gamma_node *gamma);

		void gamma_conv_nonspec(jive::gamma_node *gamma);

		void gamma_conv_spec(jive::gamma_node *gamma);

		void
		gamma_conv(jive::region *region, bool allow_speculation=true);

		void
		gamma_conv(jlm::RvsdgModule &rm, bool allow_speculation=true);
	}
}
#endif //JLM_BACKEND_HLS_RVSDG2RHLS_GAMMA_CONV_HPP

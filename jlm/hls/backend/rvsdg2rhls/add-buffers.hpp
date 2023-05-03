/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_ADD_BUFFERS_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_ADD_BUFFERS_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jlm{
	namespace hls{
		void
		add_buffers(jive::region *region, bool pass_through);

		void
		add_buffers(jlm::RvsdgModule &rm, bool pass_through);
	}
}
#endif //JLM_BACKEND_HLS_RVSDG2RHLS_ADD_BUFFERS_HPP

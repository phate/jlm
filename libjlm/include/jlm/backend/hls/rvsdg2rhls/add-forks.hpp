/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_ADD_FORKS_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_ADD_FORKS_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jlm{
	namespace hls{
		void
		add_forks(jive::region *region);

		void
		add_forks(jlm::RvsdgModule &rm);
	}
}
#endif //JLM_BACKEND_HLS_RVSDG2RHLS_ADD_FORKS_HPP

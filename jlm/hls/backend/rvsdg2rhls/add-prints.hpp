/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_ADD_PRINTS_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_ADD_PRINTS_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jlm{
	namespace hls{
		void
		add_prints(jive::region *region);

		void
		add_prints(jlm::RvsdgModule &rm);

		void
		convert_prints(jlm::RvsdgModule &rm);

		void
		convert_prints(
      jive::region *region,
      jive::output * printf,
      const FunctionType & functionType);

		jive::output *
		route_to_region(jive::output * output, jive::region * region);
	}
}

#endif //JLM_BACKEND_HLS_RVSDG2RHLS_ADD_PRINTS_HPP

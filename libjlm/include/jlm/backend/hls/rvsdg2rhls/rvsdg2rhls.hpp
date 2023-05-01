/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_RVSDG2RHLS_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_RVSDG2RHLS_HPP

#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/node.hpp>

namespace jlm{
	namespace hls{
		static inline bool is_constant(const jive::node *node) {
			return jive::is<jive::bitconstant_op>(node) ||
				   jive::is<jlm::UndefValueOperation>(node) ||
				   jive::is<jive::ctlconstant_op>(node);
		}

		void
		rvsdg2rhls(jlm::RvsdgModule &rm);

		void
		dump_ref(jlm::RvsdgModule &rhls);

        std::unique_ptr<RvsdgModule>
        split_hls_function(RvsdgModule &rm, const std::string &function_name);
    }
}
#endif //JLM_BACKEND_HLS_RVSDG2RHLS_RVSDG2RHLS_HPP

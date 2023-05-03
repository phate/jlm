/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */


#ifndef JLM_HLS_BACKEND_RHLS2FIRRTL_DOT_HLS_HPP
#define JLM_HLS_BACKEND_RHLS2FIRRTL_DOT_HLS_HPP

#include <jlm/hls/backend/rhls2firrtl/base-hls.hpp>
#include <jlm/hls/ir/hls.hpp>

namespace jlm {
	namespace hls {
		class DotHLS : public BaseHLS {
			std::string
			extension() override;

			std::string
			get_text(jlm::RvsdgModule &rm) override;

		private:
			std::string
			argument_to_dot(jive::argument *port);

			std::string
			result_to_dot(jive::result *port);

			std::string
			node_to_dot(const jive::node *node);

			std::string
			edge(std::string src, std::string snk, const jive::type &type, bool back = false);

			std::string
			loop_to_dot(hls::loop_node *ln);

			void
			prepare_loop_out_port(hls::loop_node *ln);

			std::string
			subregion_to_dot(jive::region *sr);

			int loop_ctr = 0;
		};
	}
}

#endif //JLM_HLS_BACKEND_RHLS2FIRRTL_DOT_HLS_HPP

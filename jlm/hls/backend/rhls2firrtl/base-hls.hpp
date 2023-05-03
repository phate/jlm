/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RHLS2FIRRTL_BASE_HLS_HPP
#define JLM_BACKEND_HLS_RHLS2FIRRTL_BASE_HLS_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>

#include <fstream>

namespace jlm {
	namespace hls {
		bool
		isForbiddenChar(char c);

		class BaseHLS {
		public:
			std::string
			run(jlm::RvsdgModule &rm) {
				assert(node_map.empty());
				// ensure consistent naming across runs
				create_node_names(get_hls_lambda(rm)->subregion());
				return get_text(rm);
			}

		private:
			virtual std::string
			extension() = 0;

		protected:
			std::unordered_map<const jive::node *, std::string> node_map;
			std::unordered_map<jive::output *, std::string> output_map;

			std::string
			get_node_name(const jive::node *node);

			static std::string
			get_port_name(jive::input *port);

			static std::string
			get_port_name(jive::output *port);

			const jlm::lambda::node *
			get_hls_lambda(jlm::RvsdgModule &rm);

			int
			JlmSize(const jive::type *type);

			void
			create_node_names(jive::region *r);

			virtual std::string
			get_text(jlm::RvsdgModule &rm) = 0;

			static std::string
			get_base_file_name(const RvsdgModule &rm);
		};
	}
}


#endif //JLM_BACKEND_HLS_RHLS2FIRRTL_BASE_HLS_HPP

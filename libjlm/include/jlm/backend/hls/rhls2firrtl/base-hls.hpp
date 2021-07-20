//
// Created by david on 7/5/21.
//

#ifndef JLM_BACKEND_HLS_RHLS2FIRRTL_BASE_HLS_HPP
#define JLM_BACKEND_HLS_RHLS2FIRRTL_BASE_HLS_HPP

#include <jlm/ir/RvsdgModule.hpp>
#include <fstream>
#include <jlm/ir/operators/lambda.hpp>
#include <jlm/ir/operators/operators.hpp>

namespace jlm {
	namespace hls {
		bool
		isForbiddenChar(char c);

		class BaseHLS {
		public:
			void
			run(jlm::RvsdgModule &rm) {
				assert(node_map.empty());
				// ensure consistent naming across runs
				create_node_names(get_hls_lambda(rm)->subregion());
				auto text = get_text(rm);
				std::basic_string<char, std::char_traits<char>, std::allocator<char>> base_file_name = get_base_file_name(
						rm);
				std::string file_name = base_file_name + extension();
				std::ofstream out_file;
				out_file.open(file_name);
				out_file << text;
				out_file.close();
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

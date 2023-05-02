/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RHLS2FIRRTL_FIRRTL_HLS_HPP
#define JLM_BACKEND_HLS_RHLS2FIRRTL_FIRRTL_HLS_HPP

#include <jlm/ir/hls/hls.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/sext.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/load.hpp>
#include <jlm/llvm/ir/operators/store.hpp>
#include <jlm/rvsdg/bitstring/comparison.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include "base-hls.hpp"

#include <string>


namespace jlm {
	namespace hls {
		class FirrtlModule {
		public:
			FirrtlModule(const std::string &name, const std::string &firrtl, bool hasMem) : has_mem(hasMem), name(name),
																							firrtl(firrtl) {}

		public:
			bool has_mem;
			std::string name;
			std::string firrtl;
		};


		inline bool
		is_identity_mapping(const jive::match_op &op);


		int jlm_sizeof(const jive::type * t);

		class FirrtlHLS : public BaseHLS {
			std::string
			extension() override {
				return ".fir";
			}

			std::string
			get_text(jlm::RvsdgModule &rm) override;

		private:
			std::vector<std::pair<const jive::operation*, FirrtlModule>> modules;
			std::vector<std::string> mem_nodes;

			std::string
			get_module_name(const jive::node *node);

			static inline std::string
			indent(size_t depth) {
				return std::string(depth * 4, ' ');
			}

			static std::string
			ready(jive::output *port) {
				return get_port_name(port) + ".ready";
			}

			static std::string
			ready(jive::input *port) {
				return get_port_name(port) + ".ready";
			}

			static std::string
			valid(jive::output *port) {
				return get_port_name(port) + ".valid";
			}

			static std::string
			valid(jive::input *port) {
				return get_port_name(port) + ".valid";
			}

			static std::string
			data(jive::output *port) {
				return get_port_name(port) + ".data";
			}

			static std::string
			data(jive::input *port) {
				return get_port_name(port) + ".data";
			}

			static std::string
			fire(jive::output *port) {
				return "and(" + valid(port) + ", " + ready(port) + ")";
			}

			static std::string
			fire(jive::input *port) {
				return "and(" + valid(port) + ", " + ready(port) + ")";
			}

			static std::string
			UInt(size_t width, size_t value) {
				return jive::detail::strfmt("UInt<", width, ">(", value, ")");
			}

			static std::string
			to_firrtl_type(const jive::type *type);

			std::string
			mem_io();

			std::string
			mux_mem(const std::vector<std::string> &mem_nodes) const;

			std::string
			module_header(const jive::node *node, bool has_mem_io = false);


			FirrtlModule &
			mem_node_to_firrtl(const jive::simple_node *n);

			FirrtlModule &
			pred_buffer_to_firrtl(const jive::simple_node *n);

			FirrtlModule &
			buffer_to_firrtl(const jive::simple_node *n);

			FirrtlModule &
			ndmux_to_firrtl(const jive::simple_node *n);

			FirrtlModule &
			dmux_to_firrtl(const jive::simple_node *n);

			FirrtlModule &
			merge_to_firrtl(const jive::simple_node *n);

			FirrtlModule &
			fork_to_firrtl(const jive::simple_node *n);

			FirrtlModule &
			sink_to_firrtl(const jive::simple_node *n);

			FirrtlModule &
			print_to_firrtl(const jive::simple_node *n);

			FirrtlModule &
			branch_to_firrtl(const jive::simple_node *n);

			FirrtlModule &
			trigger_to_firrtl(const jive::simple_node *n);

			static std::string
			gep_op_to_firrtl(const jive::simple_node *n);

			static std::string
			match_op_to_firrtl(const jive::simple_node *n);

			static std::string
			simple_op_to_firrtl(const jive::simple_node *n);

			FirrtlModule &
			single_out_simple_node_to_firrtl(const jive::simple_node *n);

			FirrtlModule
			node_to_firrtl(const jive::node *node, const int depth);

			std::string
			create_loop_instances(hls::loop_node *ln);

			std::string
			connect(jive::region *sr);

			FirrtlModule
			subregion_to_firrtl(jive::region *sr);

			FirrtlModule
			lambda_node_to_firrtl(const jlm::lambda::node *ln);
		};

	}
}
#endif //JLM_BACKEND_HLS_RHLS2FIRRTL_FIRRTL_HLS_HPP

/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RHLS2FIRRTL_VERILATOR_HARNESS_HLS_HPP
#define JLM_BACKEND_HLS_RHLS2FIRRTL_VERILATOR_HARNESS_HLS_HPP


#include <jlm/rvsdg/bitstring/type.hpp>
#include "base-hls.hpp"

namespace jlm {
	namespace hls {
		class VerilatorHarnessHLS : public BaseHLS {
			std::string
			extension() override {
				return "_harness.cpp";
			}

			std::string
			get_text(jlm::RvsdgModule &rm) override;

		private:
			std::string
			convert_to_c_type(const jive::type* type);

			std::string
			convert_to_c_type_postfix(const jive::type* type);
		};
	}
}

#endif //JLM_BACKEND_HLS_RHLS2FIRRTL_VERILATOR_HARNESS_HLS_HPP

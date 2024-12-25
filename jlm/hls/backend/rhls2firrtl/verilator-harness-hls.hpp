/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RHLS2FIRRTL_VERILATOR_HARNESS_HLS_HPP
#define JLM_HLS_BACKEND_RHLS2FIRRTL_VERILATOR_HARNESS_HLS_HPP

#include <jlm/hls/backend/rhls2firrtl/base-hls.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>

namespace jlm::hls
{

class VerilatorHarnessHLS : public BaseHLS
{
  const util::filepath VerilogFile_;

  std::string
  extension() override
  {
    return "_harness.cpp";
  }

  std::string
  GetText(llvm::RvsdgModule & rm) override;

public:
  /**
   * Construct a Verilator harness generator.
   *
   * @param verilogFile The filename to the Verilog file that is to be used together with the
   * generated harness as input to Verilator.
   */
  explicit VerilatorHarnessHLS(util::filepath verilogFile)
      : VerilogFile_(std::move(verilogFile))
  {}
};

}

#endif // JLM_HLS_BACKEND_RHLS2FIRRTL_VERILATOR_HARNESS_HLS_HPP

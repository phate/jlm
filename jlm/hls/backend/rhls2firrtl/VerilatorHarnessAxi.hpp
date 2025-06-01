/*
 * Copyright 2025 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RHLS2FIRRTL_VERILATORHARNESSAXI_HPP
#define JLM_HLS_BACKEND_RHLS2FIRRTL_VERILATORHARNESSAXI_HPP

#include <jlm/hls/backend/rhls2firrtl/base-hls.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>

namespace jlm::hls
{
class VerilatorHarnessAxi : public BaseHLS
{
  const util::FilePath VerilogFile_;

  std::string
  extension() override
  {
    return "harness_axi.cpp";
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
  explicit VerilatorHarnessAxi(util::FilePath verilogFile)
      : VerilogFile_(std::move(verilogFile))
  {}
};

}

#endif // JLM_HLS_BACKEND_RHLS2FIRRTL_VERILATORHARNESSAXI_HPP

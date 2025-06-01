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

std::string
ConvertToCType(const rvsdg::Type * type);

std::optional<std::string>
GetReturnTypeAsC(const rvsdg::LambdaNode & kernel);

std::tuple<size_t, std::string, std::string>
GetParameterListAsC(const rvsdg::LambdaNode & kernel);

class VerilatorHarnessHLS : public BaseHLS
{
  const util::FilePath VerilogFile_;

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
  explicit VerilatorHarnessHLS(util::FilePath verilogFile)
      : VerilogFile_(std::move(verilogFile))
  {}
};

}

#endif // JLM_HLS_BACKEND_RHLS2FIRRTL_VERILATOR_HARNESS_HLS_HPP

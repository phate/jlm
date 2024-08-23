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
  std::string
  extension() override
  {
    return "_harness.cpp";
  }

  std::string
  get_text(llvm::RvsdgModule & rm) override;

public:
  /**
   * Construct a Verilator harness generator.
   *
   * /param verilogFile The filename to the Verilog file that is to be used together with the
   * generated harness as input to Verilator.
   */
  explicit VerilatorHarnessHLS(util::filepath verilogFile)
      : VerilogFile_(std::move(verilogFile))
  {}

private:
  const util::filepath VerilogFile_;

  /**
   * \return The Verilog filename that is to be used together with the generated harness as input to
   * Verilator.
   */
  [[nodiscard]] const util::filepath &
  GetVerilogFileName() const noexcept
  {
    return VerilogFile_;
  }

  std::string
  convert_to_c_type(const jlm::rvsdg::type * type);

  std::string
  convert_to_c_type_postfix(const jlm::rvsdg::type * type);

  void
  get_function_header(
      std::ostringstream & cpp,
      const llvm::lambda::node * ln,
      const std::string & function_name);

  void
  call_function(
      std::ostringstream & cpp,
      const llvm::lambda::node * ln,
      const std::string & function_name);
};

}

#endif // JLM_HLS_BACKEND_RHLS2FIRRTL_VERILATOR_HARNESS_HLS_HPP

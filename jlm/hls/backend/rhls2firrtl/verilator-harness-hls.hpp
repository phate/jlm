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
  * /param verilogFile The filepath to the input Verilog file to Verilator
  */
  VerilatorHarnessHLS(const util::filepath verilogFile)
    : VerilogFile_(verilogFile) {};

private:
  const util::filepath VerilogFile_;

  /**
   * Get the include name.
   *
   * \return The include name.
   */
  [[nodiscard]] const std::string
  GetIncludeName() const noexcept
  {
    return VerilogFile_.base() + ".h";
  }

  /**
   * Get the V_NAME.
   *
   * \return The V_NAME.
   */
  [[nodiscard]] const std::string
  GetVName() const noexcept
  {
    return VerilogFile_.base();
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

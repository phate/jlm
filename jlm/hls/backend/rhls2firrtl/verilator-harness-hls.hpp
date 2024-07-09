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
   * The generated harness inserts an header file as an include that is used when simulating with
   * Verilator. This include must have the same name as the Verilog file that is passed as input to
   * verilator.
   *
   * \param includeFileName The base name, i.e., no extension, of the Verilog file that is to be
   *                        passed as input to Verilator.
   */
  void
  SetIncludeFileName(const std::string & includeFileName)
  {
    IncludeFileName_ = includeFileName;
  }

private:
  std::string IncludeFileName_ = "";

  /**
   * Get the include filename that has been set by SetIncludeFileName().
   *
   * \return The include filename that has been set by SetIncludeFileName().
   */
  std::string &
  GetIncludeFileName()
  {
    return IncludeFileName_;
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

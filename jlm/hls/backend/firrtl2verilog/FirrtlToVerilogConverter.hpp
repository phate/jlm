/*
 * Copyright 2024 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_FIRRTL2VERILOG_FIRRTLTOVERILOGCONVERTER_HPP
#define JLM_HLS_BACKEND_FIRRTL2VERILOG_FIRRTLTOVERILOGCONVERTER_HPP

#include <jlm/util/file.hpp>

namespace jlm::hls
{

class FirrtlToVerilogConverter
{
public:
  FirrtlToVerilogConverter() = delete;

  /**
   * Converts FIRRTL to Verilog by reading FIRRTL from a file and writing the converted Verilog to
   * another file. The functionality is heavily inspired by the processBuffer() function of the
   * firtool in the CIRCT project.
   *
   * \param inputFirrtlFile The complete path to the FIRRTL file to convert to Verilog.
   * \param outputVerilogFile The complete path to the Verilog file to write the converted Verilog
   * to. \return True if the conversion was successful, false otherwise.
   */
  static bool
  Convert(const util::FilePath inputFirrtlFile, const util::FilePath outputVerilogFile);
};

} // namespace jlm::hls

#endif // JLM_HLS_BACKEND_FIRRTL2VERILOG_FIRRTLTOVERILOGCONVERTER_HPP

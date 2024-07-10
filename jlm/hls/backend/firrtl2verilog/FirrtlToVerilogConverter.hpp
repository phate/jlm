/*
 * Copyright 2024 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_FIRRTL2VERILOG_FIRRTLTOVERILOGCONVERTER_HPP
#define JLM_HLS_BACKEND_FIRRTL2VERILOG_FIRRTLTOVERILOGCONVERTER_HPP

#include <jlm/util/file.hpp>

#include <circt/Conversion/ExportVerilog.h>
#include <circt/Dialect/FIRRTL/FIRParser.h>
#include <circt/Dialect/HW/HWOps.h>
#include <circt/Dialect/SV/SVPasses.h>
#include <circt/Firtool/Firtool.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/FileUtilities.h>

namespace jlm::hls
{

using namespace circt;
using namespace llvm;

/**
 * Converts FIRRTL to Verilog by reading FIRRTL from a file and writing the converted Verilog to
 * another file. The functionality is heavily inspired by the processBuffer() function of the
 * firtool in the CIRCT project.
 *
 * \param inputFirrtlFile The complete path to the FIRRTL file to convert to Verilog.
 * \param outputVerilogFile The complete path to the Verilog file to write the converted Verilog to.
 * \return True if the conversion was successful, false otherwise.
 */
static bool
FirrtlToVerilogConverter(
    const util::filepath inputFirrtlFile,
    const util::filepath outputVerilogFile)
{
  mlir::MLIRContext context;
  mlir::TimingScope ts;

  // Set up and read the input FIRRTL file
  std::string errorMessage;
  auto input = mlir::openInputFile(inputFirrtlFile.to_str(), &errorMessage);
  if (!input)
  {
    std::cerr << errorMessage << std::endl;
    return false;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  mlir::SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);

  firrtl::FIRParserOptions options;
  options.infoLocatorHandling = firrtl::FIRParserOptions::InfoLocHandling::IgnoreInfo;
  options.numAnnotationFiles = 0;
  options.scalarizeExtModules = true;
  auto module = importFIRFile(sourceMgr, &context, ts, options);
  if (!module)
  {
    std::cerr << "Failed to parse FIRRTL input" << std::endl;
    return false;
  }

  // Manually set up the options for the firtool
  cl::OptionCategory mainCategory("firtool Options");
  firtool::FirtoolOptions firtoolOptions(mainCategory);
  firtoolOptions.preserveAggregate = firrtl::PreserveAggregate::PreserveMode::None;
  firtoolOptions.preserveMode = firrtl::PreserveValues::PreserveMode::None;
  firtoolOptions.buildMode = firtool::FirtoolOptions::BuildModeRelease;
  firtoolOptions.exportChiselInterface = false;

  // Populate the pass manager and apply them to the module
  mlir::PassManager pm(&context);

  // Firtool sets a blackBoxRoot based on the inputFilename path, but this functionality is not used
  // so we set it to an empty string (the final argument)
  if (failed(firtool::populateCHIRRTLToLowFIRRTL(pm, firtoolOptions, *module, "")))
  {
    std::cerr << "Failed to populate CHIRRTL to LowFIRRTL" << std::endl;
    return false;
  }
  if (failed(firtool::populateLowFIRRTLToHW(pm, firtoolOptions)))
  {
    std::cerr << "Failed to populate LowFIRRTL to HW" << std::endl;
    return false;
  }
  if (failed(firtool::populateHWToSV(pm, firtoolOptions)))
  {
    std::cerr << "Failed to populate HW to SV" << std::endl;
    return false;
  }

  if (failed(pm.run(module.get())))
  {
    std::cerr << "Failed to run pass manager" << std::endl;
    return false;
  }

  mlir::PassManager exportPm(&context);

  // Legalize unsupported operations within the modules.
  exportPm.nest<hw::HWModuleOp>().addPass(sv::createHWLegalizeModulesPass());

  // Tidy up the IR to improve verilog emission quality.
  exportPm.nest<hw::HWModuleOp>().addPass(sv::createPrettifyVerilogPass());

  std::error_code errorCode;
  llvm::raw_fd_ostream os(outputVerilogFile.to_str(), errorCode);
  exportPm.addPass(createExportVerilogPass(os));

  if (failed(exportPm.run(module.get())))
  {
    std::cerr << "Failed to run export pass manager" << std::endl;
    return false;
  }

  (void)module.release();
  return true;
}

} // namespace jlm::hls

#endif // JLM_HLS_BACKEND_FIRRTL2VERILOG_FIRRTLTOVERILOGCONVERTER_HPP

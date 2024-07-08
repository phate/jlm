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
 * Converts FIRRTL to Verilog by reading FIRRTL from a file and writing the converted Verilog to another file.
 * The functionality is heavily inspired by the processBuffer() function of the firtool in the CIRCT project.
 *
 * \param inputFirrtlFile The complete path to the FIRRTL file to convert to Verilog.
 * \param outputVerilogFile The complete path to the Verilog file to write the converted Verilog to.
 */
static void
FirrtlToVerilogConverter(const std::string inputFirrtlFile, const std::string outputVerilogFile)
{
  mlir::MLIRContext context;
  mlir::TimingScope ts;

  // Set up and read the input FIRRTL file
  std::string errorMessage;
  auto input = mlir::openInputFile(inputFirrtlFile, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
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
    llvm::errs() << "Failed to parse FIRRTL input\n";
    exit(1);
  }

  // Manually set up the options for the firtool
  cl::OptionCategory mainCategory("firtool Options");
  firtool::FirtoolOptions firtoolOptions(mainCategory);
  firtoolOptions.preserveAggregate = firrtl::PreserveAggregate::PreserveMode::None;
  firtoolOptions.preserveMode = firrtl::PreserveValues::PreserveMode::None;
  firtoolOptions.buildMode = firtool::FirtoolOptions::BuildModeRelease;
  firtoolOptions.exportChiselInterface = false;
  seq::ExternalizeClockGateOptions clockOptions;
  clockOptions.moduleName = "EICG_wrapper";
  clockOptions.inputName = "in";
  clockOptions.outputName = "out";
  clockOptions.enableName = "en";
  clockOptions.testEnableName = "test_en";
  clockOptions.instName = "ckg";
  firtoolOptions.clockGateOpts = clockOptions;

  // Populate the pass manager and apply them to the module
  mlir::PassManager pm(&context);

  if (failed(firtool::populateCHIRRTLToLowFIRRTL(pm, firtoolOptions, *module, inputFirrtlFile)))
  {
    llvm::errs() << "Failed to populate CHIRRTL to LowFIRRTL\n";
    exit(1);
  }
  if (failed(firtool::populateLowFIRRTLToHW(pm, firtoolOptions)))
  {
    llvm::errs() << "Failed to populate LowFIRRTL to HW\n";
    exit(1);
  }
  if (failed(firtool::populateHWToSV(pm, firtoolOptions)))
  {
    llvm::errs() << "Failed to populate HW to SV\n";
    exit(1);
  }

  if (failed(pm.run(module.get())))
  {
    llvm::errs() << "Failed to run pass manager\n";
    exit(1);
  }

  mlir::PassManager exportPm(&context);

  // Legalize unsupported operations within the modules.
  exportPm.nest<hw::HWModuleOp>().addPass(sv::createHWLegalizeModulesPass());

  // Tidy up the IR to improve verilog emission quality.
  exportPm.nest<hw::HWModuleOp>().addPass(sv::createPrettifyVerilogPass());

  std::error_code errorCode;
  llvm::raw_fd_ostream os(outputVerilogFile, errorCode);
  exportPm.addPass(createExportVerilogPass(os));

  if (failed(exportPm.run(module.get())))
  {
    llvm::errs() << "Failed to run export pass manager\n";
    exit(1);
  }

  (void)module.release();
}

} // namespace jlm::hls

#endif // JLM_HLS_BACKEND_FIRRTL2VERILOG_FIRRTLTOVERILOGCONVERTER_HPP
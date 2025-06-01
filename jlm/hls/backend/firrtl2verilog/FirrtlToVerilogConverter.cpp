/*
 * Copyright 2024 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/firrtl2verilog/FirrtlToVerilogConverter.hpp>

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

bool
FirrtlToVerilogConverter::Convert(
    const util::FilePath inputFirrtlFile,
    const util::FilePath outputVerilogFile)
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
  options.scalarizePublicModules = true;
  options.scalarizeExtModules = true;
  auto module = importFIRFile(sourceMgr, &context, ts, options);
  if (!module)
  {
    std::cerr << "Failed to parse FIRRTL input" << std::endl;
    return false;
  }

  // Manually set the options for the firtool
  firtool::FirtoolOptions firtoolOptions;
  firtoolOptions.setOutputFilename(outputVerilogFile.to_str());
  firtoolOptions.setPreserveAggregate(firrtl::PreserveAggregate::PreserveMode::None);
  firtoolOptions.setPreserveValues(firrtl::PreserveValues::PreserveMode::Named);
  firtoolOptions.setBuildMode(firtool::FirtoolOptions::BuildModeDefault);
  firtoolOptions.setChiselInterfaceOutDirectory("");
  firtoolOptions.setDisableHoistingHWPassthrough(true);
  firtoolOptions.setOmirOutFile("");
  firtoolOptions.setBlackBoxRootPath("");
  firtoolOptions.setReplSeqMemFile("");
  firtoolOptions.setOutputAnnotationFilename("");

  // Populate the pass manager and apply them to the module
  mlir::PassManager pm(&context);
  if (failed(firtool::populatePreprocessTransforms(pm, firtoolOptions)))
  {
    std::cerr << "Failed to populate preprocess transforms" << std::endl;
    return false;
  }

  // Firtool sets a blackBoxRoot based on the inputFilename path, but this functionality is not used
  // so we set it to an empty string (the final argument)
  if (failed(firtool::populateCHIRRTLToLowFIRRTL(pm, firtoolOptions, "")))
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
  std::error_code errorCode;
  llvm::raw_fd_ostream os(outputVerilogFile.to_str(), errorCode);
  if (failed(firtool::populateExportVerilog(pm, firtoolOptions, os)))
  {
    std::cerr << "Failed to populate Export Verilog" << std::endl;
    return false;
  }

  if (failed(pm.run(module.get())))
  {
    std::cerr << "Failed to run pass manager" << std::endl;
    return false;
  }

  (void)module.release();
  return true;
}

} // namespace jlm::hls

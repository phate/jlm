/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/firrtl2verilog/FirrtlToVerilogConverter.hpp>
#include <jlm/hls/backend/rhls2firrtl/dot-hls.hpp>
#include <jlm/hls/backend/rhls2firrtl/json-hls.hpp>
#include <jlm/hls/backend/rhls2firrtl/RhlsToFirrtlConverter.hpp>
#include <jlm/hls/backend/rhls2firrtl/verilator-harness-hls.hpp>
#include <jlm/hls/backend/rhls2firrtl/VerilatorHarnessAxi.hpp>
#include <jlm/hls/backend/rvsdg2rhls/add-buffers.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/llvm/backend/IpGraphToLlvmConverter.hpp>
#include <jlm/llvm/backend/RvsdgToIpGraphConverter.hpp>
#include <jlm/llvm/DotWriter.hpp>
#include <jlm/llvm/frontend/InterProceduralGraphConversion.hpp>
#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/tooling/CommandLine.hpp>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/SourceMgr.h>

static void
stringToFile(const std::string & output, const jlm::util::FilePath & fileName)
{
  std::ofstream outputFile;
  outputFile.open(fileName.to_str());
  outputFile << output;
  outputFile.close();
}

static void
llvmToFile(jlm::llvm::RvsdgModule & module, const jlm::util::FilePath & fileName)
{
  llvm::LLVMContext ctx;
  jlm::util::StatisticsCollector statisticsCollector;
  auto jm = jlm::llvm::RvsdgToIpGraphConverter::CreateAndConvertModule(module, statisticsCollector);
  auto lm = jlm::llvm::IpGraphToLlvmConverter::CreateAndConvertModule(*jm, ctx);
  std::error_code EC;
  llvm::raw_fd_ostream os(fileName.to_str(), EC);
  lm->print(os, nullptr);
}

int
main(int argc, char ** argv)
{
  auto & commandLineOptions = jlm::tooling::JlmHlsCommandLineParser::Parse(argc, argv);

  llvm::LLVMContext ctx;
  llvm::SMDiagnostic err;
  auto llvmModule = llvm::parseIRFile(commandLineOptions.InputFile_.to_str(), err, ctx);
  llvmModule->setSourceFileName(commandLineOptions.OutputFiles_.Dirname().Join("jlm_hls").to_str());
  if (!llvmModule)
  {
    err.print(argv[0], llvm::errs());
    exit(1);
  }

  // TODO: Get demanded statistics and output folder from command line options
  auto moduleName = commandLineOptions.InputFile_.base();
  jlm::util::StatisticsCollectorSettings settings(
      {},
      jlm::util::FilePath::TempDirectoryPath(),
      moduleName);
  jlm::util::StatisticsCollector collector(std::move(settings));

  /* LLVM to JLM pass */
  auto jlmModule = jlm::llvm::ConvertLlvmModule(*llvmModule);

  auto rvsdgModule = jlm::llvm::ConvertInterProceduralGraphModule(*jlmModule, collector);

  jlm::hls::setMemoryLatency(commandLineOptions.MemoryLatency_);

  if (commandLineOptions.ExtractHlsFunction_)
  {
    auto hlsFunction = jlm::hls::split_hls_function(*rvsdgModule, commandLineOptions.HlsFunction_);

    llvmToFile(*rvsdgModule, commandLineOptions.OutputFiles_.WithSuffix(".rest.ll"));
    llvmToFile(*hlsFunction, commandLineOptions.OutputFiles_.WithSuffix(".function.ll"));
    return 0;
  }

  if (commandLineOptions.OutputFormat_
      == jlm::tooling::JlmHlsCommandLineOptions::OutputFormat::Firrtl)
  {
    jlm::hls::rvsdg2ref(*rvsdgModule, commandLineOptions.OutputFiles_.WithSuffix(".ref.ll"));

    jlm::llvm::LlvmDotWriter dotWriter;
    jlm::util::HashSet<std::unique_ptr<jlm::rvsdg::Transformation>> transformations;
    auto transformationSequence =
        jlm::hls::createTransformationSequence(dotWriter, false, transformations);
    transformationSequence->Run(*rvsdgModule, collector);

    // Writing the FIRRTL to a file and then reading it back in to convert to Verilog.
    // Could potentially change to pass the FIRRTL directly to the converter, but the converter
    // is based on CIRCT's Firtool library, which assumes that the FIRRTL is read from a file.
    jlm::hls::RhlsToFirrtlConverter hls;
    auto output = hls.ToString(*rvsdgModule);

    const auto firrtlFile = commandLineOptions.OutputFiles_.WithSuffix(".fir");
    stringToFile(output, firrtlFile);

    const auto outputVerilogFile = commandLineOptions.OutputFiles_.WithSuffix(".v");
    if (!jlm::hls::FirrtlToVerilogConverter::Convert(firrtlFile, outputVerilogFile))
    {
      std::cerr << "The FIRRTL to Verilog conversion failed.\n" << std::endl;
      exit(1);
    }

    jlm::hls::VerilatorHarnessHLS vhls(outputVerilogFile);
    stringToFile(
        vhls.run(*rvsdgModule),
        commandLineOptions.OutputFiles_.WithSuffix(".harness.cpp"));

    jlm::hls::VerilatorHarnessAxi ahls(outputVerilogFile);
    stringToFile(
        ahls.run(*rvsdgModule),
        commandLineOptions.OutputFiles_.WithSuffix(".harness_axi.cpp"));

    // TODO: hide behind flag
    jlm::hls::JsonHLS jhls;
    stringToFile(jhls.run(*rvsdgModule), commandLineOptions.OutputFiles_.WithSuffix(".json"));
  }
  else if (
      commandLineOptions.OutputFormat_ == jlm::tooling::JlmHlsCommandLineOptions::OutputFormat::Dot)
  {
    jlm::llvm::LlvmDotWriter dotWriter;
    jlm::util::HashSet<std::unique_ptr<jlm::rvsdg::Transformation>> transformations;
    auto transformationSequence =
        jlm::hls::createTransformationSequence(dotWriter, false, transformations);
    transformationSequence->Run(*rvsdgModule, collector);

    jlm::hls::DotHLS dhls;
    stringToFile(
        dhls.run(*rvsdgModule),
        commandLineOptions.OutputFiles_.Dirname().Join("jlm_hls.dot"));
  }
  else
  {
    JLM_UNREACHABLE("Format not supported.\n");
  }

  collector.PrintStatistics();

  return 0;
}

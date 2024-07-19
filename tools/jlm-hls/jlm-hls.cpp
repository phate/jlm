/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/firrtl2verilog/FirrtlToVerilogConverter.hpp>
#include <jlm/hls/backend/rhls2firrtl/dot-hls.hpp>
#include <jlm/hls/backend/rhls2firrtl/json-hls.hpp>
#include <jlm/hls/backend/rhls2firrtl/RhlsToFirrtlConverter.hpp>
#include <jlm/hls/backend/rhls2firrtl/verilator-harness-hls.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/llvm/backend/jlm2llvm/jlm2llvm.hpp>
#include <jlm/llvm/backend/rvsdg2jlm/rvsdg2jlm.hpp>
#include <jlm/llvm/frontend/InterProceduralGraphConversion.hpp>
#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/tooling/CommandLine.hpp>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/SourceMgr.h>

static void
stringToFile(std::string output, std::string fileName)
{
  std::ofstream outputFile;
  outputFile.open(fileName);
  outputFile << output;
  outputFile.close();
}

static void
llvmToFile(jlm::llvm::RvsdgModule & module, std::string fileName)
{
  llvm::LLVMContext ctx;
  jlm::util::StatisticsCollector statisticsCollector;
  auto jm = jlm::llvm::rvsdg2jlm::rvsdg2jlm(module, statisticsCollector);
  auto lm = jlm::llvm::jlm2llvm::convert(*jm, ctx);
  std::error_code EC;
  llvm::raw_fd_ostream os(fileName, EC);
  lm->print(os, nullptr);
}

int
main(int argc, char ** argv)
{
  auto & commandLineOptions = jlm::tooling::JlmHlsCommandLineParser::Parse(argc, argv);

  llvm::LLVMContext ctx;
  llvm::SMDiagnostic err;
  auto llvmModule = llvm::parseIRFile(commandLineOptions.InputFile_.to_str(), err, ctx);
  llvmModule->setSourceFileName(commandLineOptions.OutputFiles_.path() + "/jlm_hls");
  if (!llvmModule)
  {
    err.print(argv[0], llvm::errs());
    exit(1);
  }

  /* LLVM to JLM pass */
  auto jlmModule = jlm::llvm::ConvertLlvmModule(*llvmModule);
  jlm::util::StatisticsCollector statisticsCollector;
  auto rvsdgModule = jlm::llvm::ConvertInterProceduralGraphModule(*jlmModule, statisticsCollector);

  if (commandLineOptions.ExtractHlsFunction_)
  {
    auto hlsFunction = jlm::hls::split_hls_function(*rvsdgModule, commandLineOptions.HlsFunction_);

    llvmToFile(*rvsdgModule, commandLineOptions.OutputFiles_.to_str() + ".rest.ll");
    llvmToFile(*hlsFunction, commandLineOptions.OutputFiles_.to_str() + ".function.ll");
    return 0;
  }

  if (commandLineOptions.OutputFormat_
      == jlm::tooling::JlmHlsCommandLineOptions::OutputFormat::Firrtl)
  {
    jlm::hls::rvsdg2ref(*rvsdgModule, commandLineOptions.OutputFiles_.to_str() + ".ref.ll");
    jlm::hls::rvsdg2rhls(*rvsdgModule);

    // Writing the FIRRTL to a file and then reading it back in to convert to Verilog.
    // Could potentially change to pass the FIRRTL directly to the converter, but the converter
    // is based on CIRCT's Firtool library, which assumes that the FIRRTL is read from a file.
    jlm::hls::RhlsToFirrtlConverter hls;
    auto output = hls.ToString(*rvsdgModule);
    jlm::util::filepath firrtlFile(commandLineOptions.OutputFiles_.to_str() + ".fir");
    stringToFile(output, firrtlFile.to_str());
    jlm::util::filepath outputVerilogFile(commandLineOptions.OutputFiles_.to_str() + ".v");
    if (!jlm::hls::FirrtlToVerilogConverter::Convert(firrtlFile, outputVerilogFile))
    {
      std::cerr << "The FIRRTL to Verilog conversion failed.\n" << std::endl;
      exit(1);
    }

    jlm::hls::VerilatorHarnessHLS vhls(outputVerilogFile);
    stringToFile(vhls.run(*rvsdgModule), commandLineOptions.OutputFiles_.to_str() + ".harness.cpp");

    // TODO: hide behind flag
    jlm::hls::JsonHLS jhls;
    stringToFile(jhls.run(*rvsdgModule), commandLineOptions.OutputFiles_.to_str() + ".json");
  }
  else if (
      commandLineOptions.OutputFormat_ == jlm::tooling::JlmHlsCommandLineOptions::OutputFormat::Dot)
  {
    jlm::hls::rvsdg2rhls(*rvsdgModule);

    jlm::hls::DotHLS dhls;
    stringToFile(dhls.run(*rvsdgModule), commandLineOptions.OutputFiles_.path() + "/jlm_hls.dot");
  }
  else
  {
    JLM_UNREACHABLE("Format not supported.\n");
  }

  return 0;
}

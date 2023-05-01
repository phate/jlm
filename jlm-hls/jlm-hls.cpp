/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/backend/jlm2llvm/jlm2llvm.hpp>
#include <jlm/llvm/backend/rvsdg2jlm/rvsdg2jlm.hpp>
#include <jlm/backend/hls/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/backend/hls/rhls2firrtl/dot-hls.hpp>
#include <jlm/backend/hls/rhls2firrtl/firrtl-hls.hpp>
#include <jlm/backend/hls/rhls2firrtl/mlirgen.hpp>
#include <jlm/backend/hls/rhls2firrtl/verilator-harness-hls.hpp>
#include <jlm/llvm/frontend/InterProceduralGraphConversion.hpp>
#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/tooling/CommandLine.hpp>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/SourceMgr.h>


static void
stringToFile(
	std::string output,
	std::string fileName)
{
	std::ofstream outputFile;
	outputFile.open(fileName);
	outputFile << output;
	outputFile.close();
}

static void
llvmToFile(
	jlm::RvsdgModule &module,
	std::string fileName)
{
	llvm::LLVMContext ctx;
	jlm::StatisticsCollector statisticsCollector;
	auto jm = jlm::rvsdg2jlm::rvsdg2jlm(module, statisticsCollector);
	auto lm = jlm::jlm2llvm::convert(*jm, ctx);
	std::error_code EC;
	llvm::raw_fd_ostream os(fileName, EC);
	lm->print(os, nullptr);
}

int
main(int argc, char ** argv)
{
  auto & commandLineOptions = jlm::JlmHlsCommandLineParser::Parse(argc, argv);

	llvm::LLVMContext ctx;
	llvm::SMDiagnostic err;
	auto llvmModule = llvm::parseIRFile(commandLineOptions.InputFile_.to_str(), err, ctx);
	llvmModule->setSourceFileName(commandLineOptions.OutputFolder_.path() + "/jlm_hls");
	if (!llvmModule) {
		err.print(argv[0], llvm::errs());
		exit(1);
	}

	/* LLVM to JLM pass */
	auto jlmModule = jlm::ConvertLlvmModule(*llvmModule);
	jlm::StatisticsCollector statisticsCollector;
	auto rvsdgModule = jlm::ConvertInterProceduralGraphModule(
    *jlmModule,
    statisticsCollector);

	if (commandLineOptions.ExtractHlsFunction_) {
		auto hlsFunction = jlm::hls::split_hls_function(
					*rvsdgModule,
					commandLineOptions.HlsFunction_);

		llvmToFile(
			*rvsdgModule,
      commandLineOptions.OutputFolder_.path() + "/jlm_hls_rest.ll");
		llvmToFile(
			*hlsFunction,
      commandLineOptions.OutputFolder_.path() + "/jlm_hls_function.ll");
		return 0;
	}

	if (commandLineOptions.OutputFormat_ == jlm::JlmHlsCommandLineOptions::OutputFormat::Firrtl) {
		jlm::hls::rvsdg2rhls(*rvsdgModule);

		std::string output;
		if (commandLineOptions.UseCirct_) {
			jlm::hls::MLIRGen hls;
			output = hls.run(*rvsdgModule);
		} else {
			jlm::hls::FirrtlHLS hls;
			output = hls.run(*rvsdgModule);
		}
		stringToFile(
			output,
      commandLineOptions.OutputFolder_.path() + "/jlm_hls.fir");

		jlm::hls::VerilatorHarnessHLS vhls;
		stringToFile(
			vhls.run(*rvsdgModule),
      commandLineOptions.OutputFolder_.path() + "/jlm_hls_harness.cpp");
	} else if (commandLineOptions.OutputFormat_ == jlm::JlmHlsCommandLineOptions::OutputFormat::Dot) {
		jlm::hls::rvsdg2rhls(*rvsdgModule);

		jlm::hls::DotHLS dhls;
		stringToFile(
			dhls.run(*rvsdgModule),
      commandLineOptions.OutputFolder_.path() + "/jlm_hls.dot");
	} else {
		JLM_UNREACHABLE("Format not supported.\n");
	}
	return 0;
}

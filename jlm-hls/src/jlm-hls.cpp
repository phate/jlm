/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/view.hpp>

#include <jlm/backend/llvm/jlm2llvm/jlm2llvm.hpp>
#include <jlm/backend/llvm/rvsdg2jlm/rvsdg2jlm.hpp>
#include <jlm/backend/hls/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/backend/hls/rhls2firrtl/dot-hls.hpp>
#include <jlm/backend/hls/rhls2firrtl/firrtl-hls.hpp>
#include <jlm/backend/hls/rhls2firrtl/mlirgen.hpp>
#include <jlm/backend/hls/rhls2firrtl/verilator-harness-hls.hpp>
#include <jlm/frontend/llvm/InterProceduralGraphConversion.hpp>
#include <jlm/frontend/llvm/LlvmModuleConversion.hpp>
#include <jlm/ir/RvsdgModule.hpp>
#include <jlm/util/Statistics.hpp>

#include <jlm-hls/cmdline.hpp>

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
	jlm::StatisticsDescriptor sd;
	auto jm = jlm::rvsdg2jlm::rvsdg2jlm(module, sd);
	auto lm = jlm::jlm2llvm::convert(*jm, ctx);
	std::error_code EC;
	llvm::raw_fd_ostream os(fileName, EC);
	lm->print(os, nullptr);
}

int
main(int argc, char ** argv)
{
	jlm::JlmHlsCommandLineOptions flags;
	parse_cmdline(argc, argv, flags);

	llvm::LLVMContext ctx;
	llvm::SMDiagnostic err;
	auto llvmModule = llvm::parseIRFile(flags.inputFile.to_str(), err, ctx);
	llvmModule->setSourceFileName(flags.outputFolder.path() + "/jlm_hls");
	if (!llvmModule) {
		err.print(argv[0], llvm::errs());
		exit(1);
	}

	/* LLVM to JLM pass */
	auto jlmModule = jlm::ConvertLlvmModule(*llvmModule);
	jlm::StatisticsDescriptor sd;
	auto rvsdgModule = jlm::ConvertInterProceduralGraphModule(
					*jlmModule,
					sd);

	if (flags.extractHlsFunction) {
		auto hlsFunction = jlm::hls::split_hls_function(
					*rvsdgModule,
					flags.hlsFunction);

		llvmToFile(
			*rvsdgModule,
			flags.outputFolder.path() + "/jlm_hls_rest.ll");
		llvmToFile(
			*hlsFunction,
			flags.outputFolder.path() + "/jlm_hls_function.ll");
		return 0;
	}

	if (flags.format == jlm::OutputFormat::firrtl) {
		jlm::hls::rvsdg2rhls(*rvsdgModule);

		std::string output;
		if (flags.useCirct) {
			jlm::hls::MLIRGen hls;
			output = hls.run(*rvsdgModule);
		} else {
			jlm::hls::FirrtlHLS hls;
			output = hls.run(*rvsdgModule);
		}
		stringToFile(
			output,
			flags.outputFolder.path() + "/jlm_hls.fir");

		jlm::hls::VerilatorHarnessHLS vhls;
		stringToFile(
			vhls.run(*rvsdgModule),
			flags.outputFolder.path() + "/jlm_hls_harness.cpp");
	} else if (flags.format == jlm::OutputFormat::dot) {
		jlm::hls::rvsdg2rhls(*rvsdgModule);

		jlm::hls::DotHLS dhls;
		stringToFile(
			dhls.run(*rvsdgModule),
			flags.outputFolder.path() + "/jlm_hls.dot");
	} else {
		JLM_UNREACHABLE("Format not supported.\n");
	}
	return 0;
}

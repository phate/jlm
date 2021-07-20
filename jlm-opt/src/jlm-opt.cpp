/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
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
#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/RvsdgModule.hpp>
#include <jlm/opt/optimization.hpp>

#include <jlm-opt/cmdline.hpp>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include <iostream>
#include <unistd.h>

static std::unique_ptr<llvm::Module>
parse_llvm_file(
	const char * executable,
	const jlm::filepath & file,
	llvm::LLVMContext & ctx)
{
	llvm::SMDiagnostic d;
	auto module = llvm::parseIRFile(file.to_str(), d, ctx);
	if (!module) {
		d.print(executable, llvm::errs());
		exit(EXIT_FAILURE);
	}

	return module;
}

static std::unique_ptr<jlm::ipgraph_module>
construct_jlm_module(llvm::Module & module)
{
	return jlm::ConvertLlvmModule(module);
}

static void
print_as_xml(
	const jlm::RvsdgModule & rm,
	const jlm::filepath & fp,
	const jlm::StatisticsDescriptor&)
{
	auto fd = fp == "" ? stdout : fopen(fp.to_str().c_str(), "w");

	jive::view_xml(rm.Rvsdg().root(), fd);

	if (fd != stdout)
			fclose(fd);
}

static void
print_as_llvm(
	const jlm::RvsdgModule & rm,
	const jlm::filepath & fp,
	const jlm::StatisticsDescriptor & sd)
{
	auto jlm_module = jlm::rvsdg2jlm::rvsdg2jlm(rm, sd);

	llvm::LLVMContext ctx;
	auto llvm_module = jlm::jlm2llvm::convert(*jlm_module, ctx);

	if (fp == "") {
		llvm::raw_os_ostream os(std::cout);
		llvm_module->print(os, nullptr);
	} else {
		std::error_code ec;
		llvm::raw_fd_ostream os(fp.to_str(), ec);
		llvm_module->print(os, nullptr);
	}
}

static void
print(
	const jlm::RvsdgModule & rm,
	const jlm::filepath & fp,
	const jlm::outputformat & format,
	const jlm::StatisticsDescriptor & sd)
{
	using namespace jlm;

	static std::unordered_map<
		jlm::outputformat,
		std::function<void(const RvsdgModule&, const filepath&, const StatisticsDescriptor&)>
	> formatters({
		{outputformat::xml,  print_as_xml}
	, {outputformat::llvm, print_as_llvm}
	});

	JLM_ASSERT(formatters.find(format) != formatters.end());
	formatters[format](rm, fp, sd);
}

static void
dump_xml(
	 std::unique_ptr<jlm::RvsdgModule> &rvsdgModule,
	 const std::string &suffix = "")
{
	auto source_file_name = rvsdgModule->SourceFileName().name();
	std::string file_name = source_file_name.substr(
				0,
				source_file_name.find_last_of('.')) + suffix + ".rvsdg";
	auto xml_file = fopen(file_name.c_str(), "w");
	jive::view_xml(rvsdgModule->Rvsdg().root(), xml_file);
	fclose(xml_file);
}

static void
dump_llvm(
	  jlm::RvsdgModule &module,
	  std::string outfile)
{
	llvm::LLVMContext ctx;
	jlm::StatisticsDescriptor sd;
	auto jm = jlm::rvsdg2jlm::rvsdg2jlm(module, sd);
	auto lm = jlm::jlm2llvm::convert(*jm, ctx);
	std::error_code EC;
	llvm::raw_fd_ostream os(outfile, EC);
	lm->print(os, nullptr);
}

int
main(int argc, char ** argv)
{
	jlm::cmdline_options flags;
	parse_cmdline(argc, argv, flags);

	llvm::LLVMContext ctx;
	if (flags.hls_function != "") {
		llvm::SMDiagnostic err;
		auto llvmModule = llvm::parseIRFile(flags.hls_file, err, ctx);
		// change folder to redirect output
		if(!flags.outfolder.empty()){
			assert(chdir(flags.outfolder.c_str())==0);
		}
		if (!llvmModule) {
			err.print(argv[0], llvm::errs());
			exit(1);
		}

		/* LLVM to JLM pass */
		auto jlmModule = jlm::ConvertLlvmModule(*llvmModule);
		auto rvsdgModule = jlm::ConvertInterProceduralGraphModule(
					*jlmModule,
					flags.sd);

		auto hlsFunction = jlm::hls::split_hls_function(
					*rvsdgModule,
					flags.hls_function);

		dump_llvm(*rvsdgModule, "jlm_hls.rest.ll");
		dump_llvm(*hlsFunction, "jlm_hls.function.ll");

		return 0;
	} else if (flags.hls_file != "") {
		llvm::SMDiagnostic err;
		auto llvmModule = llvm::parseIRFile(flags.hls_file, err, ctx);
		// change folder to redirect output
		if(!flags.outfolder.empty()){
			assert(chdir(flags.outfolder.c_str())==0);
		}
		// TODO: fix this hack
		llvmModule->setSourceFileName(/*ff.path()+*/"jlm_hls");
		if (!llvmModule) {
			err.print(argv[0], llvm::errs());
			exit(1);
		}

		/* LLVM to JLM pass */
		auto jlmModule = jlm::ConvertLlvmModule(*llvmModule);
		auto rhls = jlm::ConvertInterProceduralGraphModule(*jlmModule, flags.sd);

		dump_xml(rhls);
		jlm::hls::rvsdg2rhls(*rhls);

		jlm::hls::MLIRGen hls;
		hls.run(*rhls);

		if (flags.circt) {
			jlm::hls::MLIRGen hls;
			hls.run(*rhls);
		} else {
			jlm::hls::FirrtlHLS hls;
			hls.run(*rhls);
		}

		jlm::hls::DotHLS dhls;
		dhls.run(*rhls);
		jlm::hls::VerilatorHarnessHLS vhls;
		vhls.run(*rhls);

		return 0;
	}

	auto llvm_module = parse_llvm_file(argv[0], flags.ifile, ctx);

	auto jlm_module = construct_jlm_module(*llvm_module);

	llvm_module.reset();
	auto rvsdgModule = jlm::ConvertInterProceduralGraphModule(*jlm_module, flags.sd);

	optimize(*rvsdgModule, flags.sd, flags.optimizations);

	print(*rvsdgModule, flags.ofile, flags.format, flags.sd);

	return 0;
}

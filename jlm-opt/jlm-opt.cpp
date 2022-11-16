/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/view.hpp>

#include <jlm/backend/llvm/jlm2llvm/jlm2llvm.hpp>
#include <jlm/backend/llvm/rvsdg2jlm/rvsdg2jlm.hpp>
#include <jlm/frontend/llvm/InterProceduralGraphConversion.hpp>
#include <jlm/frontend/llvm/LlvmModuleConversion.hpp>
#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/RvsdgModule.hpp>
#include <jlm/opt/optimization.hpp>
#include <jlm/tooling/CommandLine.hpp>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include <iostream>

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
	jlm::StatisticsCollector&)
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
	jlm::StatisticsCollector & statisticsCollector)
{
	auto jlm_module = jlm::rvsdg2jlm::rvsdg2jlm(rm, statisticsCollector);

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
	const jlm::JlmOptCommandLineOptions::OutputFormat & format,
	jlm::StatisticsCollector & statisticsCollector)
{
  using namespace jlm;

  static std::unordered_map<
    jlm::JlmOptCommandLineOptions::OutputFormat,
    std::function<void(const RvsdgModule&, const filepath&, StatisticsCollector&)>
  > formatters(
    {
      {JlmOptCommandLineOptions::OutputFormat::Xml,  print_as_xml},
      {JlmOptCommandLineOptions::OutputFormat::Llvm, print_as_llvm}
    });

  JLM_ASSERT(formatters.find(format) != formatters.end());
  formatters[format](rm, fp, statisticsCollector);
}

int
main(int argc, char ** argv)
{
  auto & commandLineOptions = jlm::JlmOptCommandLineParser::Parse(argc, argv);

  jlm::StatisticsCollector statisticsCollector(commandLineOptions.StatisticsCollectorSettings_);

  llvm::LLVMContext llvmContext;
  auto llvmModule = parse_llvm_file(
    argv[0],
    commandLineOptions.InputFile_,
    llvmContext);

  auto interProceduralGraphModule = construct_jlm_module(*llvmModule);
  llvmModule.reset();

  auto rvsdgModule = jlm::ConvertInterProceduralGraphModule(
    *interProceduralGraphModule,
    statisticsCollector);

  optimize(
    *rvsdgModule,
    statisticsCollector,
    commandLineOptions.Optimizations_);

  print(
    *rvsdgModule,
    commandLineOptions.OutputFile_,
    commandLineOptions.OutputFormat_,
    statisticsCollector);

  statisticsCollector.PrintStatistics();

  return 0;
}

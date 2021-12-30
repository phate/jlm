/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/view.hpp>

#include <jlm/backend/llvm/jlm2llvm/jlm2llvm.hpp>
#include <jlm/backend/llvm/rvsdg2jlm/rvsdg2jlm.hpp>
#include <jlm/frontend/llvm/jlm2rvsdg/module.hpp>
#include <jlm/frontend/llvm/llvm2jlm/module.hpp>
#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/opt/optimization.hpp>

#include <jlm-opt/cmdline.hpp>

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
	return jlm::convert_module(module);
}

static void
print_as_xml(
	const jlm::rvsdg_module & rm,
	const jlm::filepath & fp,
	const jlm::StatisticsDescriptor&)
{
	auto fd = fp == "" ? stdout : fopen(fp.to_str().c_str(), "w");

	jive::view_xml(rm.graph()->root(), fd);

	if (fd != stdout)
			fclose(fd);
}

static void
print_as_llvm(
	const jlm::rvsdg_module & rm,
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
	const jlm::rvsdg_module & rm,
	const jlm::filepath & fp,
	const jlm::outputformat & format,
	const jlm::StatisticsDescriptor & sd)
{
	using namespace jlm;

	static std::unordered_map<
		jlm::outputformat,
		std::function<void(const rvsdg_module&, const filepath&, const StatisticsDescriptor&)>
	> formatters({
		{outputformat::xml,  print_as_xml}
	, {outputformat::llvm, print_as_llvm}
	});

	JLM_ASSERT(formatters.find(format) != formatters.end());
	formatters[format](rm, fp, sd);
}

int
main(int argc, char ** argv)
{
	jlm::cmdline_options flags;
	parse_cmdline(argc, argv, flags);

	llvm::LLVMContext ctx;
	auto llvm_module = parse_llvm_file(argv[0], flags.ifile, ctx);

	auto jlm_module = construct_jlm_module(*llvm_module);

	llvm_module.reset();
	auto rm = jlm::construct_rvsdg(*jlm_module, flags.sd);

	optimize(*rm, flags.sd, flags.optimizations);

	print(*rm, flags.ofile, flags.format, flags.sd);

	return 0;
}

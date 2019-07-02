/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/view.h>

#include <jlm/ir/module.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/rvsdg.hpp>
#include <jlm/jlm2rvsdg/module.hpp>
#include <jlm/jlm2llvm/jlm2llvm.hpp>
#include <jlm/llvm2jlm/module.hpp>
#include <jlm/opt/optimization.hpp>
#include <jlm/rvsdg2jlm/rvsdg2jlm.hpp>

#include <jlm-opt/cmdline.hpp>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include <chrono>
#include <iostream>

int
main(int argc, char ** argv)
{
	jlm::cmdline_options flags;
	parse_cmdline(argc, argv, flags);

	llvm::SMDiagnostic d;
	llvm::LLVMContext ctx;
	auto lm = llvm::parseIRFile(flags.ifile.to_str(), d, ctx);
	if (!lm) {
		d.print(argv[0], llvm::errs());
		exit(1);
	}

	auto jm = jlm::convert_module(*lm);

	#ifdef RVSDGTIME
		size_t ntacs = jlm::ntacs(*jm);
		auto start = std::chrono::high_resolution_clock::now();
	#endif

	auto rvsdg = jlm::construct_rvsdg(*jm);

	#ifdef RVSDGTIME
		size_t nnodes = jive::nnodes(rvsdg->graph()->root());
	#endif

	for (const auto & opt : flags.optimizations)
		optimize(*rvsdg->graph(), opt);

	if (flags.format == jlm::outputformat::llvm) {
		jm = jlm::rvsdg2jlm::rvsdg2jlm(*rvsdg);

		#ifdef RVSDGTIME
			auto end = std::chrono::high_resolution_clock::now();
			std::cerr << "RVSDGTIME: "
			          << ntacs << " "
			          << nnodes << " "
			          << std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()
			          << "\n";
		#endif

		lm = jlm::jlm2llvm::convert(*jm, ctx);
		llvm::raw_os_ostream os(std::cout);
		lm->print(os, nullptr);
	}

	if (flags.format == jlm::outputformat::xml)
		jive::view_xml(rvsdg->graph()->root(), stdout);

	return 0;
}

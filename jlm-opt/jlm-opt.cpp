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

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include <chrono>
#include <iostream>

struct cmdflags {
	inline
	cmdflags()
	: xml(false)
	, llvm(false)
	{}

	bool xml;
	bool llvm;
	std::vector<jlm::optimization> passes;
};

static void
print_usage(const std::string & app)
{
	std::cerr << "Usage: " << app << " [OPTIONS] FILE\n";
	std::cerr << "OPTIONS:\n";
	std::cerr << "--cne: Perform common node elimination.\n";
	std::cerr << "--dne: Perform dead node elimination.\n";
	std::cerr << "--iln: Perform function inlining.\n";
	std::cerr << "--inv: Perform invariant value redirection.\n";
	std::cerr << "--pll: Perform node pull in.\n";
	std::cerr << "--psh: Perform node push out.\n";
	std::cerr << "--red: Perform node reductions.\n";
	std::cerr << "--ivt: Perform theta-gamma inversion.\n";
	std::cerr << "--url: Perform loop unrolling.\n";
	std::cerr << "--llvm: Output LLVM IR.\n";
	std::cerr << "--xml: Output RVSDG as XML.\n";
}

static std::string
parse_cmdflags(int argc, char ** argv, cmdflags & flags)
{
	if (argc < 2) {
		std::cerr << "Expected LLVM IR file as input.\n";
		print_usage(argv[0]);
		exit(1);
	}

	static std::unordered_map<std::string, void(*)(cmdflags&)> map({
	  {"--cne", [](cmdflags & flags){ flags.passes.push_back(jlm::optimization::cne); }}
	, {"--dne", [](cmdflags & flags){ flags.passes.push_back(jlm::optimization::dne); }}
	, {"--iln", [](cmdflags & flags){ flags.passes.push_back(jlm::optimization::iln); }}
	, {"--inv", [](cmdflags & flags){ flags.passes.push_back(jlm::optimization::inv); }}
	, {"--pll", [](cmdflags & flags){ flags.passes.push_back(jlm::optimization::pll); }}
	, {"--psh", [](cmdflags & flags){ flags.passes.push_back(jlm::optimization::psh); }}
	, {"--red", [](cmdflags & flags){ flags.passes.push_back(jlm::optimization::red); }}
	, {"--ivt", [](cmdflags & flags){ flags.passes.push_back(jlm::optimization::ivt); }}
	, {"--url", [](cmdflags & flags){ flags.passes.push_back(jlm::optimization::url); }}
	, {"--llvm", [](cmdflags & flags){ flags.llvm = true; }}
	, {"--xml", [](cmdflags & flags){ flags.xml = true; }}
	});

	for (int n = 1; n < argc-1; n++) {
		std::string flag(argv[n]);
		if (map.find(flag) != map.end()) {
			map[flag](flags);
			continue;
		}

		std::cerr << "Unknown command line flag: " << flag << "\n";
		print_usage(argv[0]);
		exit(1);
	}

	return std::string(argv[argc-1]);
}

int
main(int argc, char ** argv)
{
	cmdflags flags;
	auto file = parse_cmdflags(argc, argv, flags);

	llvm::SMDiagnostic d;
	llvm::LLVMContext ctx;
	auto lm = llvm::parseIRFile(file, d, ctx);
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

	for (const auto & opt : flags.passes)
		optimize(*rvsdg->graph(), opt);

	if (flags.llvm) {
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

	if (flags.xml)
		jive::view_xml(rvsdg->graph()->root(), stdout);

	return 0;
}

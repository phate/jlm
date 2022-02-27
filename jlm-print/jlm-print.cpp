/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/backend/llvm/jlm2llvm/jlm2llvm.hpp>
#include <jlm/backend/llvm/rvsdg2jlm/rvsdg2jlm.hpp>
#include "jlm/frontend/llvm/LlvmModuleConversion.hpp"
#include <jlm/frontend/llvm/jlm2rvsdg/InterProceduralGraphConversion.hpp>
#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/print.hpp>
#include <jlm/ir/RvsdgModule.hpp>
#include <jlm/util/Statistics.hpp>

#include <jive/view.hpp>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include <iostream>

#include <getopt.h>

class cmdflags {
public:
	inline
	cmdflags()
		: l2j(false)
		, j2r(false)
		, r2j(false)
		, j2l(false)
		, j2rx(false)
		, l2jdot(false)
		, l2j_ipg_dot(false)
		, r2jdot(false)
		, r2j_ipg_dot(false)
	{}

	bool l2j;
	bool j2r;
	bool r2j;
	bool j2l;
	bool j2rx;
	bool l2jdot;
	bool l2j_ipg_dot;
	bool r2jdot;
	bool r2j_ipg_dot;
	std::string file;
	jlm::StatisticsDescriptor sd;
	std::string l2jdot_function;
	std::string r2jdot_function;
};

static void
print_usage(const std::string & app)
{
	std::cerr << "Usage: " << app << " [OPTIONS]\n";
	std::cerr << "OPTIONS:\n";
	std::cerr << "--l2j: Print program after LLVM to JLM pass.\n";
	std::cerr << "--j2r: Print program after JLM to RVSDG pass.\n";
	std::cerr << "--r2j: Print program after RVSDG to JLM pass.\n";
	std::cerr << "--j2l: Print program after JLM to LLVM pass.\n";
	std::cerr << "--j2rx: Print program as XML after JLM to RVSDG pass.\n";
	std::cerr << "--l2jdot f: Print function f as graphviz after LLVM to JLM pass.\n";
	std::cerr << "--l2j-ipg-dot: Print inter-procedure graph after LLVM to JLM pass.\n";
	std::cerr << "--r2jdot f: Print function f as graphviz after RVSDG to JLM pass.\n";
	std::cerr << "--r2j-ipg-dot: Print inter-procedure graph after RVSDG to JLM pass.\n";
	std::cerr << "--file name: LLVM IR file.\n";
}

static void
parse_cmdflags(int argc, char ** argv, cmdflags & flags)
{
	static constexpr size_t l2j = 1;
	static constexpr size_t j2r = 2;
	static constexpr size_t r2j = 3;
	static constexpr size_t j2l = 4;
	static constexpr size_t j2rx = 5;
	static constexpr size_t l2jdot = 6;
	static constexpr size_t l2j_ipg_dot = 7;
	static constexpr size_t r2jdot = 8;
	static constexpr size_t r2j_ipg_dot = 9;
	static constexpr size_t file = 10;

	static struct option options[] = {
	  {"l2j", no_argument, NULL, l2j}
	, {"j2r", no_argument, NULL, j2r}
	, {"r2j", no_argument, NULL, r2j}
	, {"j2l", no_argument, NULL, j2l}
	, {"j2rx", no_argument, NULL, j2rx}
	, {"l2jdot", required_argument, NULL, l2jdot}
	, {"l2j-ipg-dot", no_argument, NULL, l2j_ipg_dot}
	, {"r2jdot", required_argument, NULL, r2jdot}
	, {"r2j-ipg-dot", no_argument, NULL, r2j_ipg_dot}
	, {"file", required_argument, NULL, file}
	, {NULL, 0, NULL, 0}
	};

	int opt;
	while ((opt = getopt_long_only(argc, argv, "", options, NULL)) != -1) {
		switch (opt) {
			case l2j: { flags.l2j = true; break; }
			case j2r: { flags.j2r = true; break; }
			case r2j: { flags.r2j = true; break; }
			case j2l: { flags.j2l = true; break; }
			case j2rx: { flags.j2rx = true; break; }
			case l2j_ipg_dot: { flags.l2j_ipg_dot = true; break; }
			case r2j_ipg_dot: { flags.r2j_ipg_dot = true; break; }

			case l2jdot: { flags.l2jdot = true; flags.l2jdot_function = optarg; break; }
			case r2jdot: { flags.r2jdot = true; flags.r2jdot_function = optarg; break; }

			case file: { flags.file = optarg; break; }

			default:
				print_usage(argv[0]);
				exit(EXIT_FAILURE);
		}
	}

	if (flags.file.empty()) {
		print_usage(argv[0]);
		exit(EXIT_FAILURE);
	}
}

static inline const jlm::cfg *
find_cfg(
	const jlm::ipgraph & ipg,
	const std::string & name)
{
	const jlm::ipgraph_node * node = nullptr;
	for (auto & n: ipg) {
		if (n.name() == name)
			node = &n;
	}

	if (!node) {
		std::cerr << "Function " << name << " not found.\n";
		exit(1);
	}

	auto f = dynamic_cast<const jlm::function_node*>(node);
	if (!f) {
		std::cerr << "Function " << name << " not found.\n";
		exit(1);
	}

	if (!f->cfg()) {
		std::cerr << "Function " << name << " has no CFG.\n";
		exit(1);
	}

	return f->cfg();
}

int
main (int argc, char ** argv)
{
	cmdflags flags;
	parse_cmdflags(argc, argv, flags);

	llvm::LLVMContext ctx;
	llvm::SMDiagnostic err;
	auto lm = llvm::parseIRFile(flags.file, err, ctx);

	if (!lm) {
		err.print(argv[0], llvm::errs());
		exit(1);
	}

	/* LLVM to JLM pass */
	auto jm = jlm::convert_module(*lm);
	if (flags.l2j)
		jlm::print(*jm, stdout);
	if (flags.l2jdot)
		jlm::print_dot(*find_cfg(jm->ipgraph(), flags.l2jdot_function), stdout);
	if (flags.l2j_ipg_dot)
		jlm::print_dot(jm->ipgraph(), stdout);

	auto rvsdgModule = jlm::ConvertInterProceduralGraphModule(*jm, flags.sd);
	if (flags.j2r) jive::view(rvsdgModule->Rvsdg().root(), stdout);
	if (flags.j2rx) jive::view_xml(rvsdgModule->Rvsdg().root(), stdout);

	jm = jlm::rvsdg2jlm::rvsdg2jlm(*rvsdgModule, flags.sd);
	if (flags.r2j)
		jlm::print(*jm, stdout);
	if (flags.r2jdot)
		jlm::print_dot(*find_cfg(jm->ipgraph(), flags.r2jdot_function), stdout);
	if (flags.r2j_ipg_dot)
		jlm::print_dot(jm->ipgraph(), stdout);

	lm = jlm::jlm2llvm::convert(*jm, ctx);
	if (flags.j2l) {
		llvm::raw_os_ostream os(std::cout);
		lm->print(os, nullptr);
	}

	return 0;
}

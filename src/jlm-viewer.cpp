/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/module.hpp>
#include <jlm/ir/rvsdg.hpp>
#include <jlm/ir/view.hpp>
#include <jlm/llvm2jlm/module.hpp>
#include <jlm/jlm2llvm/jlm2llvm.hpp>
#include <jlm/jlm2rvsdg/module.hpp>
#include <jlm/rvsdg2jlm/rvsdg2jlm.hpp>

#include <jive/rvsdg/graph.h>
#include <jive/view.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include <iostream>

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
		, r2jdot(false)
	{}

	bool l2j;
	bool j2r;
	bool r2j;
	bool j2l;
	bool j2rx;
	bool l2jdot;
	bool r2jdot;
	std::string l2jdot_function;
	std::string r2jdot_function;
};

static void
print_usage(const std::string & app)
{
	std::cerr << "Usage: " << app << " [OPTIONS] FILE\n";
	std::cerr << "OPTIONS:\n";
	std::cerr << "--l2j: Print program after LLVM to JLM pass.\n";
	std::cerr << "--j2r: Print program after JLM to RVSDG pass.\n";
	std::cerr << "--r2j: Print program after RVSDG to JLM pass.\n";
	std::cerr << "--j2l: Print program after JLM to LLVM pass.\n";
	std::cerr << "--j2rx: Print program as XML after JLM to RVSDG pass.\n";
	std::cerr << "--l2jdot f: Print function f as graphviz after LLVM to JLM pass.\n";
	std::cerr << "--r2jdot f: Print function f as graphviz after RVSDG to JLM pass.\n";
}

std::string
parse_cmdflags(int argc, char ** argv, cmdflags & cmdf)
{
	if (argc < 2) {
		std::cerr << "Expected LLVM IR file as input.\n";
		print_usage(argv[0]);
		exit(1);
	}

	for (int n = 1; n < argc-1; n++) {
		std::string flag(argv[n]);

		if (flag == "--l2j") {
			cmdf.l2j = true;
			continue;
		}

		if (flag == "--j2r") {
			cmdf.j2r = true;
			continue;
		}

		if (flag == "--j2rx") {
			cmdf.j2rx = true;
			continue;
		}

		if (flag == "--r2j") {
			cmdf.r2j = true;
			continue;
		}

		if (flag == "--j2l") {
			cmdf.j2l = true;
			continue;
		}

		if (flag == "--l2jdot") {
			cmdf.l2jdot = true;
			if (n+1 == argc-1) {
				std::cerr << "Expected LLVM IR file as input.\n";
				exit(1);
			}

			cmdf.l2jdot_function = std::string(argv[++n]);
			continue;
		}

		if (flag == "--r2jdot") {
			cmdf.r2jdot = true;
			if (n+1 == argc-1) {
				std::cerr << "Expected LLVM IR file as input.\n";
				exit(1);
			}

			cmdf.r2jdot_function = std::string(argv[++n]);
			continue;
		}
	}

	return std::string(argv[argc-1]);
}

static inline const jlm::cfg *
find_cfg(
	const jlm::callgraph & clg,
	const std::string & name)
{
	auto f = clg.lookup_function(name);
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
	auto file_name = parse_cmdflags(argc, argv, flags);

	llvm::LLVMContext ctx;
	llvm::SMDiagnostic err;
	auto lm = llvm::parseIRFile(file_name, err, ctx);

	if (!lm) {
		err.print(argv[0], llvm::errs());
		exit(1);
	}

	auto jm = jlm::convert_module(*lm);
	if (flags.l2j) jlm::view(*jm, stdout);
	if (flags.l2jdot) jlm::view_dot(*find_cfg(jm->callgraph(), flags.l2jdot_function), stdout);

	auto rvsdg = jlm::construct_rvsdg(*jm);
	if (flags.j2r) jive::view(rvsdg->graph()->root(), stdout);
	if (flags.j2rx) jive::view_xml(rvsdg->graph()->root(), stdout);

	jm = jlm::rvsdg2jlm::rvsdg2jlm(*rvsdg);
	if (flags.r2j) jlm::view(*jm, stdout);
	if (flags.r2jdot) jlm::view_dot(*find_cfg(jm->callgraph(), flags.r2jdot_function), stdout);

	lm = jlm::jlm2llvm::convert(*jm, ctx);
	if (flags.j2l) lm->dump();

	return 0;
}

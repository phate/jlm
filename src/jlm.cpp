/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/construction/module.hpp>
#include <jlm/destruction/destruction.hpp>
#include <jlm/IR/module.hpp>
#include <jlm/jlm2llvm/jlm2llvm.hpp>
#include <jlm/rvsdg2jlm/rvsdg2jlm.hpp>

#include <jive/vsdg/graph.h>
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
		: cfg(std::make_pair(false, ""))
		, clg(false)
		, llvm(false)
		, rtree(false)
		, rvsdg(false)
	{}

	std::pair<bool, std::string> cfg;
	bool clg;
	bool llvm;
	bool rtree;
	bool rvsdg;
};

std::string
parse_cmdflags(int argc, char ** argv, cmdflags & cmdf)
{
	if (argc < 2) {
		std::cerr << "Expected LLVM IR file as input\n";
		exit(1);
	}

	for (int n = 1; n < argc-1; n++) {
		std::string flag(argv[n]);

		if (flag == "-llvm") {
			cmdf.llvm = true;
			continue;
		}

		if (flag == "-clg") {
			cmdf.clg = true;
			continue;
		}

		if (flag == "-rvsdg") {
			cmdf.rvsdg = true;
			continue;
		}

		if (flag == "-rtree") {
			cmdf.rtree = true;
			continue;
		}

		if (flag == "-cfg") {
			cmdf.cfg = std::make_pair(true, std::string(argv[++n]));
			continue;
		}
	}

	return std::string(argv[argc-1]);
}

int main (int argc, char ** argv)
{
	cmdflags cmdf;
	std::string file_name = parse_cmdflags(argc, argv, cmdf);

	llvm::LLVMContext & context = llvm::getGlobalContext();

	llvm::SMDiagnostic err;
	std::unique_ptr<llvm::Module> module = llvm::parseIRFile(file_name, err, context);

	if (!module) {
		err.print(argv[0], llvm::errs());
		exit(1);
	}

	auto m = jlm::convert_module(*module);

	if (cmdf.clg)
		std::cout << m->clg().to_string();

	if (cmdf.cfg.first) {
		jlm::clg_node * f = m->clg().lookup_function(cmdf.cfg.second);
		if (!f) {
			std::cerr << "Function not found.\n";
			exit(1);
		}

		if (f->cfg())
			jive_cfg_view(*f->cfg());
	}

	auto rvsdg = jlm::construct_rvsdg(*m);

	if (cmdf.rvsdg) {
		jive::view(rvsdg->root(), stdout);
	}

	if (cmdf.rtree)
		jive::region_tree(rvsdg->root(), stdout);

	m = jlm::rvsdg2jlm::rvsdg2jlm(*rvsdg);
	module = jlm::jlm2llvm::convert(*m, context);

	if (cmdf.llvm)
		module->dump();

	return 0;
}

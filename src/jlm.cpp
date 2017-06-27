/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/module.hpp>
#include <jlm/ir/view.hpp>
#include <jlm/llvm2jlm/module.hpp>
#include <jlm/jlm2llvm/jlm2llvm.hpp>
#include <jlm/jlm2rvsdg/module.hpp>
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
		: l2j(false)
		, j2r(false)
		, r2j(false)
		, j2l(false)
	{}

	bool l2j;
	bool j2r;
	bool r2j;
	bool j2l;
};

std::string
parse_cmdflags(int argc, char ** argv, cmdflags & cmdf)
{
	if (argc < 2) {
		std::cerr << "Expected LLVM IR file as input.\n";
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

		if (flag == "--r2j") {
			cmdf.r2j = true;
			continue;
		}

		if (flag == "--j2l") {
			cmdf.j2l = true;
			continue;
		}
	}

	return std::string(argv[argc-1]);
}

int
main (int argc, char ** argv)
{
	cmdflags flags;
	auto file_name = parse_cmdflags(argc, argv, flags);

	llvm::SMDiagnostic err;
	auto lm = llvm::parseIRFile(file_name, err, llvm::getGlobalContext());

	if (!lm) {
		err.print(argv[0], llvm::errs());
		exit(1);
	}

	auto jm = jlm::convert_module(*lm);
	if (flags.l2j) jlm::view(*jm, stdout);

	auto rvsdg = jlm::construct_rvsdg(*jm);
	if (flags.j2r) jive::view(rvsdg->root(), stdout);

	jm = jlm::rvsdg2jlm::rvsdg2jlm(*rvsdg);
	if (flags.r2j) jlm::view(*jm, stdout);

	lm = jlm::jlm2llvm::convert(*jm, llvm::getGlobalContext());
	if (flags.j2l) lm->dump();

	return 0;
}

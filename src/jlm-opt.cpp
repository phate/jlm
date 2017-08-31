/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/module.hpp>
#include <jlm/ir/rvsdg.hpp>
#include <jlm/jlm2rvsdg/module.hpp>
#include <jlm/jlm2llvm/jlm2llvm.hpp>
#include <jlm/llvm2jlm/module.hpp>
#include <jlm/rvsdg2jlm/rvsdg2jlm.hpp>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>

#include <iostream>

static void
print_usage(const std::string & app)
{
	std::cerr << "Usage: " << app << " FILE\n";
}

int
main(int argc, char ** argv)
{
	if (argc != 2) {
		std::cerr << "Expected LLVM IR file as input.\n";
		print_usage(argv[0]);
		exit(1);
	}

	llvm::SMDiagnostic d;
	auto lm = llvm::parseIRFile(argv[1], d, llvm::getGlobalContext());

	auto jm = jlm::convert_module(*lm);
	auto rvsdg = jlm::construct_rvsdg(*jm);
	jm = jlm::rvsdg2jlm::rvsdg2jlm(*rvsdg);
	lm = jlm::jlm2llvm::convert(*jm, llvm::getGlobalContext());

	lm->dump();

	return 0;
}

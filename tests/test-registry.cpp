/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/frontend/clg.hpp>

#include <jlm/jlm.hpp>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include <assert.h>
#include <memory>
#include <unordered_map>

namespace jlm {

class unit_test {
public:
	unit_test(int(*v)(jlm::frontend::clg & clg)) : verify(v) {}
	int (*verify)(jlm::frontend::clg & clg);
};

static std::unordered_map<std::string, std::unique_ptr<unit_test>> unit_test_map;

void
register_unit_test(const char * name, int (*verify)(jlm::frontend::clg & clg))
{
	assert(unit_test_map.find(name) == unit_test_map.end());

	unit_test_map.insert(std::make_pair(name, std::unique_ptr<unit_test>(new unit_test(verify))));
}

int
run_unit_test(const char * name)
{
	assert(unit_test_map.find(name) != unit_test_map.end());

	std::string llname("tests/");
	llname.append(name).append(".ll");

	llvm::LLVMContext & context = llvm::getGlobalContext();

	llvm::SMDiagnostic err;
	std::unique_ptr<llvm::Module> module(llvm::ParseIRFile(llname.c_str(), err, context));
	if (!module) {
		err.print(llname.c_str(), llvm::errs());
		assert(0);
	}

	jlm::frontend::clg clg;
	convert_module(*module, clg);

	return unit_test_map[name]->verify(clg);
}

}

/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/frontend/clg.h>

#include <jlm/jlm.hpp>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include <assert.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace jlm {

class unit_test {
public:
	unit_test(const char * m, int(*v)(jive::frontend::clg & clg)) : module(m), verify(v) {}

	std::string module;
	int (*verify)(jive::frontend::clg & clg);
};

static std::unordered_map<std::string, std::unique_ptr<unit_test>> unit_test_map;

void
register_unit_test(const char * name, const char * module, int (*verify)(jive::frontend::clg & clg))
{
	assert(unit_test_map.find(std::string(name)) == unit_test_map.end());

	unit_test_map.insert(std::make_pair(std::string(name),
		std::unique_ptr<unit_test>(new unit_test(module, verify))));
}

int
run_unit_test(const char * name)
{
	assert(unit_test_map.find(std::string(name)) != unit_test_map.end());

	FILE * f = fopen("/tmp/test.c", "w");
	assert(f);
	fputs(unit_test_map[name]->module.c_str(), f);
	fclose(f);

	system("rm -rf /tmp/test.s");
	assert(!system("clang -S -emit-llvm -o /tmp/test.s /tmp/test.c"));

	llvm::LLVMContext & context = llvm::getGlobalContext();

	llvm::SMDiagnostic err;
	std::unique_ptr<llvm::Module> module(llvm::ParseIRFile("/tmp/test.s", err, context));
	if (!module) {
		err.print("/tmp/test.s", llvm::errs());
		assert(0);
	}

	jive::frontend::clg clg;
	convert_module(*module, clg);

	return unit_test_map[std::string(name)]->verify(clg);
}

}

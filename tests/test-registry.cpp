/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/vsdg/graph.h>

#include <jlm/IR/module.hpp>

#include <jlm/construction/module.hpp>
#include <jlm/destruction/destruction.hpp>

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
	unit_test(int (*v)())
	: verify(v)
	{}

	int (*verify)();
};

static std::unordered_map<std::string, std::unique_ptr<unit_test>> unit_test_map;

void
register_unit_test(const std::string & name, int (*verify)())
{
	assert(unit_test_map.find(name) == unit_test_map.end());
	unit_test_map.insert(std::make_pair(name,std::make_unique<unit_test>(verify)));
}

int
run_unit_test(const std::string & name)
{
	assert(unit_test_map.find(name) != unit_test_map.end());
	return unit_test_map[name]->verify();
}

}

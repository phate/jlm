/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_TEST_UTIL_HPP
#define JLM_TEST_UTIL_HPP

#include <llvm/IR/Module.h>
#include <llvm/Support/raw_os_ostream.h>

#include <iostream>

namespace jlm::tests
{

static inline void
print(const llvm::Module & module)
{
  llvm::raw_os_ostream os(std::cout);
  module.print(os, NULL);
}

}

#endif

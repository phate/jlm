/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_TEST_REGISTRY_HPP
#define JLM_TEST_REGISTRY_HPP

#include <string>

namespace jlm {

class module;

void
register_unit_test(const std::string & name, int(*verify)());

int
run_unit_test(const std::string & name);

}

#define JLM_UNIT_TEST_REGISTER(name, verify) \
	static void __attribute__((constructor)) register_##verify(void) \
	{ \
		jlm::register_unit_test(name, verify); \
	} \

#endif

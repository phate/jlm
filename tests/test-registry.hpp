/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_TEST_REGISTRY_HPP
#define JLM_TEST_REGISTRY_HPP

#include <string>

struct jive_graph;

namespace jlm {

class module;

void
register_unit_test(
	const std::string & name,
	int (*verify_module)(const jlm::module & m),
	int (*verify_rvsdg)(const struct jive_graph * graph));

int
run_unit_test(const std::string & name);

}

#define JLM_UNIT_TEST_REGISTER(name, verify_module, verify_rvsdg) \
	static void __attribute__((constructor)) register_##verification(void) \
	{ \
		jlm::register_unit_test(name, verify_module, verify_rvsdg); \
	} \

#endif

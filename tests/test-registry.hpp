/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_TEST_REGISTRY_HPP
#define JLM_TEST_REGISTRY_HPP

namespace jive {
namespace frontend {
	class clg;
}
}

namespace jlm {

void
register_unit_test(const char * name, const char * module, int (*verify)(jive::frontend::clg & clg));

int
run_unit_test(const char * name);

}

#define JLM_UNIT_TEST_REGISTER(name, module, verification) \
	static void __attribute__((constructor)) register_##verification(void) \
	{ \
		jlm::register_unit_test(name, module, verification); \
	} \

#endif

/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

static const char * program =
"\
	int linear() \
	{ \
		return 7; \
	} \
";

static int
verify(jive::frontend::clg & clg)
{
	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-linear", program, verify);

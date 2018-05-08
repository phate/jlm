/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/file.hpp>

#include <assert.h>

static int
test()
{
	jlm::file f("/tmp/archive.tar.gz");

	assert(f.name() == "archive.tar.gz");
	assert(f.base() == "archive");
	assert(f.complete_suffix() == "tar.gz");

	return 0;
}

JLM_UNIT_TEST_REGISTER("util/test-file", test)

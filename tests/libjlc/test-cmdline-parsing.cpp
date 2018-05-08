/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/jlc/cmdline.hpp>

#include <assert.h>
#include <string.h>

static std::vector<char*>
create_array(const std::vector<std::string> & strs)
{
	std::vector<char*> array;
	for (const auto & str : strs) {
		array.push_back(new char[str.size()+1]);
		strncpy(array.back(), str.data(), str.size());
		array.back()[str.size()] = '\0';
	}

	return array;
}

static void
destroy_array(const std::vector<char*> & array)
{
	for (const auto & ptr : array)
		delete[] ptr;
}

static void
test1()
{
	auto argv = create_array({"jlm", "-c", "-o", "foo.o", "foo.c"});

	jlm::cmdline_options cloptions;
	jlm::parse_cmdline(argv.size(), &argv[0], cloptions);

	assert(cloptions.no_linking);
	assert(cloptions.ofile.name() == "foo.o");

	destroy_array(argv);
}

static int
test()
{
	test1();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlc/test-cmdline-parsing", test)

/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/jlc/cmdline.hpp>

#include <assert.h>
#include <string.h>

static void
parse_cmdline(
	const std::vector<std::string> & args,
	jlm::cmdline_options & options)
{
	std::vector<char*> array;
	for (const auto & arg : args) {
		array.push_back(new char[arg.size()+1]);
		strncpy(array.back(), arg.data(), arg.size());
		array.back()[arg.size()] = '\0';
	}

	jlm::parse_cmdline(array.size(), &array[0], options);

	for (const auto & ptr : array)
		delete[] ptr;
}

static void
test1()
{
	jlm::cmdline_options cloptions;
	parse_cmdline({"jlc", "-c", "-o", "foo.o", "foo.c"}, cloptions);

	assert(cloptions.enable_linker == false);
	assert(cloptions.ofile.name() == "foo.o");
}

static void
test2()
{
	jlm::cmdline_options options;
	parse_cmdline({"jlc", "-o", "foobar", "/tmp/f1.o"}, options);

	assert(options.enable_parser == false);
	assert(options.enable_optimizer == false);
	assert(options.enable_assembler == false);
	assert(options.enable_linker = true);
}

static int
test()
{
	test1();
	test2();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlc/test-cmdline-parsing", test)

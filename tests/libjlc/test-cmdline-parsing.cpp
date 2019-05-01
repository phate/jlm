/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlc/cmdline.hpp>

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
	jlm::cmdline_options options;
	parse_cmdline({"jlc", "-c", "-o", "foo.o", "foo.c"}, options);

	assert(options.compilations.size() == 1);

	auto & c = options.compilations[0];
	assert(c.link() == false);
	assert(c.ofile() == "foo.o");
}

static void
test2()
{
	jlm::cmdline_options options;
	parse_cmdline({"jlc", "-o", "foobar", "/tmp/f1.o"}, options);

	assert(options.compilations.size() == 1);
	assert(options.lnkofile == "foobar");

	auto & c = options.compilations[0];
	assert(c.parse() == false);
	assert(c.optimize() == false);
	assert(c.assemble() == false);
	assert(c.link() == true);
}

static void
test3()
{
	jlm::cmdline_options options;
	parse_cmdline({"jlc", "-O", "foobar.c"}, options);

	assert(options.Olvl == jlm::optlvl::O0);
}

static void
test4()
{
	jlm::cmdline_options options;
	parse_cmdline({"jlc", "foobar.c", "-c"}, options);

	assert(options.compilations.size() == 1);

	auto & c = options.compilations[0];
	assert(c.link() == false);
	assert(c.ofile() == "foobar.o");
}

static int
test()
{
	test1();
	test2();
	test3();
	test4();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlc/test-cmdline-parsing", test)

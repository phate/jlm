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
	jlm::JlcCommandLineOptions & commandLineOptions)
{
	std::vector<char*> array;
	for (const auto & arg : args) {
		array.push_back(new char[arg.size()+1]);
		strncpy(array.back(), arg.data(), arg.size());
		array.back()[arg.size()] = '\0';
	}

	jlm::parse_cmdline(array.size(), &array[0], commandLineOptions);

	for (const auto & ptr : array)
		delete[] ptr;
}

static void
test1()
{
	jlm::JlcCommandLineOptions commandLineOptions;
	parse_cmdline({"jlc", "-c", "-o", "foo.o", "foo.c"}, commandLineOptions);

	assert(commandLineOptions.Compilations_.size() == 1);

	auto & c = commandLineOptions.Compilations_[0];
	assert(c.RequiresLinking() == false);
	assert(c.OutputFile() == "foo.o");
}

static void
test2()
{
	jlm::JlcCommandLineOptions commandLineOptions;
	parse_cmdline({"jlc", "-o", "foobar", "/tmp/f1.o"}, commandLineOptions);

	assert(commandLineOptions.Compilations_.size() == 1);
	assert(commandLineOptions.OutputFile_ == "foobar");

	auto & c = commandLineOptions.Compilations_[0];
	assert(c.RequiresParsing() == false);
	assert(c.RequiresOptimization() == false);
	assert(c.RequiresAssembly() == false);
	assert(c.RequiresLinking() == true);
}

static void
test3()
{
	jlm::JlcCommandLineOptions commandLineOptions;
	parse_cmdline({"jlc", "-O", "foobar.c"}, commandLineOptions);

	assert(commandLineOptions.OptimizationLevel_ == jlm::JlcCommandLineOptions::OptimizationLevel::O0);
}

static void
test4()
{
	jlm::JlcCommandLineOptions commandLineOptions;
	parse_cmdline({"jlc", "foobar.c", "-c"}, commandLineOptions);

	assert(commandLineOptions.Compilations_.size() == 1);

	auto & c = commandLineOptions.Compilations_[0];
	assert(c.RequiresLinking() == false);
	assert(c.OutputFile() == "foobar.o");
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

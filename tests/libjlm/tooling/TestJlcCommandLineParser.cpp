/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include "jlm/tooling/CommandLine.hpp"

#include <cassert>
#include <cstring>

static const jlm::JlcCommandLineOptions &
ParseCommandLineArguments(const std::vector<std::string> & commandLineArguments)
{
	std::vector<char*> array;
	for (const auto & commandLineArgument : commandLineArguments) {
		array.push_back(new char[commandLineArgument.size() + 1]);
		strncpy(array.back(), commandLineArgument.data(), commandLineArgument.size());
		array.back()[commandLineArgument.size()] = '\0';
	}

  static jlm::JlcCommandLineParser commandLineParser;
  auto & commandLineOptions = commandLineParser.ParseCommandLineArguments(
    static_cast<int>(array.size()),
    &array[0]);

	for (const auto & ptr : array)
		delete[] ptr;

  return commandLineOptions;
}

static void
test1()
{
  auto & commandLineOptions = ParseCommandLineArguments({"jlc", "-c", "-o", "foo.o", "foo.c"});

	assert(commandLineOptions.Compilations_.size() == 1);

	auto & c = commandLineOptions.Compilations_[0];
	assert(c.RequiresLinking() == false);
	assert(c.OutputFile() == "foo.o");
}

static void
test2()
{
  auto & commandLineOptions = ParseCommandLineArguments({"jlc", "-o", "foobar", "/tmp/f1.o"});

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
  auto & commandLineOptions = ParseCommandLineArguments({"jlc", "-O", "foobar.c"});

	assert(commandLineOptions.OptimizationLevel_ == jlm::JlcCommandLineOptions::OptimizationLevel::O0);
}

static void
test4()
{
  auto & commandLineOptions = ParseCommandLineArguments({"jlc", "foobar.c", "-c"});

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

JLM_UNIT_TEST_REGISTER("libjlm/tooling/TestJlcCommandLineParser", test)

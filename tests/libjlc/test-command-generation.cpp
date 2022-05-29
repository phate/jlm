/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlc/cmdline.hpp>
#include <jlc/command.hpp>

#include <assert.h>

static void
test1()
{
	jlm::JlcCommandLineOptions commandLineOptions;
	commandLineOptions.Compilations_.push_back({
		{"foo.c"},
		{"foo.d"},
		{"foo.o"},
		"foo.o",
		true,
		true,
		true,
		false});

	auto pgraph = jlm::generate_commands(commandLineOptions);

	auto & node = (*pgraph->GetExitNode().IncomingEdges().begin()).GetSource();
	auto cmd = dynamic_cast<const jlm::LlcCommand*>(&node.GetCommand());
	assert(cmd && cmd->OutputFile() == "foo.o");
}

static void
test2()
{
	jlm::JlcCommandLineOptions commandLineOptions;
	commandLineOptions.Compilations_.push_back({
    {"foo.o"},
    {""},
    {"foo.o"},
    "foo.o",
    false,
    false,
    false,
    true});
  commandLineOptions.OutputFile_ = {"foobar"};

	auto pgraph = jlm::generate_commands(commandLineOptions);
	assert(pgraph->NumNodes() == 3);

	auto & node = (*pgraph->GetExitNode().IncomingEdges().begin()).GetSource();
	auto cmd = dynamic_cast<const jlm::ClangCommand*>(&node.GetCommand());
	assert(cmd->InputFiles()[0] == "foo.o" && cmd->OutputFile() == "foobar");
}

static int
test()
{
	test1();
	test2();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlc/test-command-generation", test)

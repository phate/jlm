/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/jlc/cmdline.hpp>
#include <jlm/jlc/command.hpp>

#include <assert.h>

static void
test1()
{
	jlm::cmdline_options options;
	options.compilations.push_back({{"foo.c"}, {"foo.o"}, true, true, true, false});

	auto pgraph = jlm::generate_commands(options);

	auto node = (*pgraph->exit()->begin_inedges())->source();
	auto cmd = dynamic_cast<const jlm::cgencmd*>(&node->cmd());
	assert(cmd && cmd->ofile() == "foo.o");
}

static void
test2()
{
	jlm::cmdline_options options;
	options.compilations.push_back({{"foo.o"}, {"foo.o"}, false, false, false, true});
	options.lnkofile = {"foobar"};

	auto pgraph = jlm::generate_commands(options);
	assert(pgraph->nnodes() == 3);

	auto node = (*pgraph->exit()->begin_inedges())->source();
	auto cmd = dynamic_cast<const jlm::lnkcmd*>(&node->cmd());
	assert(cmd->ifiles()[0] == "foo.o" && cmd->ofile() == "foobar");
}

static int
test()
{
	test1();
	test2();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlc/test-command-generation", test)

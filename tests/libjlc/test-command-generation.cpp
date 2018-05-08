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
	jlm::cmdline_options clopts;
	clopts.no_linking = true;
	clopts.ofile = {"foo.o"};
	clopts.ifiles.push_back({"foo.c"});

	auto cmds = jlm::generate_commands(clopts);

	auto cgen = dynamic_cast<const jlm::cgencmd*>(cmds.back().get());
	assert(cgen && cgen->ofile() == "foo.o");
}

static int
test()
{
	test1();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlc/test-command-generation", test)

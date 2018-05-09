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
	clopts.enable_linker = false;
	clopts.ofile = {"foo.o"};
	clopts.ifiles.push_back({"foo.c"});

	auto cmds = jlm::generate_commands(clopts);

	auto cgen = dynamic_cast<const jlm::cgencmd*>(cmds.back().get());
	assert(cgen && cgen->ofile() == "foo.o");
}

static void
test2()
{
	jlm::cmdline_options options;
	options.enable_parser = false;
	options.enable_optimizer = false;
	options.enable_assembler = false;
	options.enable_linker = true;
	options.ofile = {"foobar"};
	options.ifiles.push_back({"foo.o"});

	auto cmds = jlm::generate_commands(options);
	assert(cmds.size() == 1);

	auto lnk = dynamic_cast<const jlm::lnkcmd*>(cmds.back().get());
	assert(lnk->ifiles()[0] == "foo.o" && lnk->ofile() == "foobar");
}

static int
test()
{
	test1();
	test2();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlc/test-command-generation", test)

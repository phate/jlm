/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/jlm/cmdline.hpp>

#include <llvm/Support/CommandLine.h>

namespace jlm {

void
parse_cmdline(int argc, char ** argv, jlm::cmdline_options & flags)
{
	using namespace llvm;

	auto & map = cl::getRegisteredOptions();
	auto & help_hidden = map["help-hidden"];
	auto & help = map["help"];
	map.clear();
	map["help"] = help;
	map["help-hidden"] = help_hidden;

	static cl::list<std::string> ifilepaths(
	  cl::Positional
	, cl::OneOrMore
	, cl::desc("<inputs>")
	);

	static cl::list<std::string> includepaths(
	  "I"
	, cl::Prefix
	, cl::ZeroOrMore
	, cl::desc("Add directory <dir> to include search paths.")
	, cl::value_desc("dir")
	);

	static cl::opt<std::string> ofilepath(
	  "o"
	, cl::init("a.out")
	, cl::desc("Write output to <file>.")
	, cl::value_desc("file")
	);

	static cl::opt<bool> only_print_commands(
	  "###"
	, cl::desc("Print (but do not run) the commands for this compilation.")
	);

	cl::ParseCommandLineOptions(argc, argv);

	flags.ofilepath = ofilepath;
	flags.ifilepaths = ifilepaths;
	flags.includepaths = includepaths;
	flags.only_print_commands = only_print_commands;
}

}

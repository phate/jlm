/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/jlm/cmdline.hpp>

#include <llvm/Support/CommandLine.h>

#include <iostream>
#include <unordered_map>

namespace jlm {

/* cmdline_options */

std::string
to_str(const optlvl & ol)
{
	static std::unordered_map<optlvl, const char*> map({
	  {optlvl::O0, "O0"}, {optlvl::O1, "O1"}
	, {optlvl::O2, "O2"}, {optlvl::O3, "O3"}
	});

	JLM_DEBUG_ASSERT(map.find(ol) != map.end());
	return map[ol];
}

/* cmdline parser */

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

	static cl::list<std::string> libpaths(
	  "L"
	, cl::Prefix
	, cl::ZeroOrMore
	, cl::desc("Add directory <dir> to library search paths.")
	, cl::value_desc("dir")
	);

	static cl::list<std::string> libs(
	  "l"
	, cl::Prefix
	, cl::ZeroOrMore
	, cl::desc("Search the library <lib> when linking.")
	, cl::value_desc("lib")
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

	static cl::opt<bool> generate_debug_information(
	  "g"
	, cl::desc("Generate source-level debug information.")
	);

	static cl::opt<char> optlvl(
	  "O"
	, cl::Prefix
	, cl::ZeroOrMore
	, cl::desc("Optimization level. [O0, O1, O2, O3]")
	, cl::value_desc("#")
	);

	static cl::list<std::string> Dmacros(
	  "D"
	, cl::Prefix
	, cl::ZeroOrMore
	, cl::desc("Add <macro> to preprocessor macros.")
	, cl::value_desc("macro")
	);

	static cl::list<std::string> Wwarnings(
	  "W"
	, cl::Prefix
	, cl::ZeroOrMore
	, cl::desc("Enable specified warning.")
	, cl::value_desc("warning")
	);

	cl::ParseCommandLineOptions(argc, argv);

	/* Process parsed options */

	static std::unordered_map<char, jlm::optlvl> Olvlmap({
	  {'0', optlvl::O0}, {'1', optlvl::O1}
	, {'2', optlvl::O2}, {'3', optlvl::O3}}
	);

	auto olvl = Olvlmap.find(optlvl);
	if (olvl == Olvlmap.end()) {
		std::cerr << "Unknown optimization level.\n";
		exit(EXIT_FAILURE);
	}

	flags.libs = libs;
	flags.macros = Dmacros;
	flags.Olvl = olvl->second;
	flags.libpaths = libpaths;
	flags.warnings = Wwarnings;
	flags.ofilepath = ofilepath;
	flags.ifilepaths = ifilepaths;
	flags.includepaths = includepaths;
	flags.only_print_commands = only_print_commands;
	flags.generate_debug_information = generate_debug_information;
}

}

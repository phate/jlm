/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/jlc/cmdline.hpp>

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

std::string
to_str(const standard & std)
{
	static std::unordered_map<standard, const char*> map({
	  {standard::none, ""}, {standard::c89, "c89"}
	, {standard::c99, "c99"}, {standard::c11, "c11"}
	, {standard::cpp98, "c++98"}, {standard::cpp03, "c++03"}
	, {standard::cpp11, "c++11"}, {standard::cpp14, "c++14"}
	});

	JLM_DEBUG_ASSERT(map.find(std) != map.end());
	return map[std];
}

/* cmdline parser */

static bool
is_objfile(const jlm::file & file)
{
	return file.complete_suffix() == "o";
}

void
parse_cmdline(int argc, char ** argv, jlm::cmdline_options & flags)
{
	using namespace llvm;

	/*
		FIXME: The command line parser setup is currently redone
		for every invocation of parse_cmdline. We should be able
		to do it only once and then reset the parser on every
		invocation of parse_cmdline.
	*/

	cl::TopLevelSubCommand->reset();

	cl::opt<bool> show_help(
	  "help"
	, cl::ValueDisallowed
	, cl::desc("Display available options."));

	cl::opt<bool> print_commands(
	  "###"
	, cl::ValueDisallowed
	, cl::desc("Print (but do not run) the commands for this compilation."));

	cl::list<std::string> ifiles(
	  cl::Positional
	, cl::desc("<inputs>"));

	cl::list<std::string> includepaths(
	  "I"
	, cl::Prefix
	, cl::desc("Add directory <dir> to include search paths.")
	, cl::value_desc("dir"));

	cl::list<std::string> libpaths(
	  "L"
	, cl::Prefix
	, cl::desc("Add directory <dir> to library search paths.")
	, cl::value_desc("dir"));

	cl::list<std::string> libs(
	  "l"
	, cl::Prefix
	, cl::desc("Search the library <lib> when linking.")
	, cl::value_desc("lib"));

	cl::opt<std::string> ofilepath(
	  "o"
	, cl::init("a.out")
	, cl::desc("Write output to <file>.")
	, cl::value_desc("file"));

	cl::opt<bool> generate_debug_information(
	  "g"
	, cl::ValueDisallowed
	, cl::desc("Generate source-level debug information."));

	cl::opt<bool> no_linking(
	  "c"
	, cl::ValueDisallowed
	, cl::desc("Only run preprocess, compile, and assemble steps."));

	cl::opt<char> optlvl(
	  "O"
	, cl::Prefix
	, cl::ZeroOrMore
	, cl::init(':')
	, cl::desc("Optimization level. [O0, O1, O2, O3]")
	, cl::value_desc("#"));

	cl::list<std::string> Dmacros(
	  "D"
	, cl::Prefix
	, cl::desc("Add <macro> to preprocessor macros.")
	, cl::value_desc("macro"));

	cl::list<std::string> Wwarnings(
	  "W"
	, cl::Prefix
	, cl::desc("Enable specified warning.")
	, cl::value_desc("warning"));

	cl::opt<std::string> std(
	  "std"
	, cl::desc("Language standard.")
	, cl::value_desc("standard"));

	cl::ParseCommandLineOptions(argc, argv);

	if (show_help)
		cl::PrintHelpMessage();

	/* Process parsed options */

	static std::unordered_map<char, jlm::optlvl> Olvlmap({
	  {'0', optlvl::O0}, {'1', optlvl::O1}
	, {'2', optlvl::O2}, {'3', optlvl::O3}}
	);

	static std::unordered_map<std::string, standard> stdmap({
		{"c89", standard::c89}, {"c90", standard::c99}
	, {"c99", standard::c99}, {"c11", standard::c11}
	, {"c++98", standard::cpp98}, {"c++03", standard::cpp03}
	, {"c++11", standard::cpp11}, {"c++14", standard::cpp14}
	});

	if (optlvl != ':') {
		auto olvl = Olvlmap.find(optlvl);
		if (olvl == Olvlmap.end()) {
			std::cerr << "Unknown optimization level.\n";
			exit(EXIT_FAILURE);
		}
		flags.Olvl = olvl->second;
	}

	if (!std.empty()) {
		auto stdit = stdmap.find(std);
		if (stdit == stdmap.end()) {
			std::cerr << "Unknown language standard.\n";
			exit(EXIT_FAILURE);
		}
		flags.std = stdit->second;
	}

	if (ifiles.empty()) {
		std::cerr << "jlc: no input files.\n";
		exit(EXIT_FAILURE);
	}

	flags.libs = libs;
	flags.macros = Dmacros;
	flags.libpaths = libpaths;
	flags.warnings = Wwarnings;
	flags.ofile = ofilepath;
	flags.enable_linker = !no_linking;

	for (const auto & ifile : ifiles)
		flags.ifiles.push_back({ifile});

	JLM_DEBUG_ASSERT(!flags.ifiles.empty());
	if (is_objfile(flags.ifiles[0])) {
		flags.enable_parser = false;
		flags.enable_optimizer = false;
		flags.enable_assembler = false;
	}

	flags.includepaths = includepaths;
	flags.only_print_commands = print_commands;
	flags.generate_debug_information = generate_debug_information;
}

}

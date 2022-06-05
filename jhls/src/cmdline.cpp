/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jhls/cmdline.hpp>

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

	JLM_ASSERT(map.find(ol) != map.end());
	return map[ol];
}

std::string
to_str(const standard & std)
{
	static std::unordered_map<standard, const char*> map({
	  {standard::none, ""}
	, {standard::gnu89, "gnu89"}, {standard::gnu99, "gnu99"}
	, {standard::c89, "c89"}, {standard::c99, "c99"}, {standard::c11, "c11"}
	, {standard::cpp98, "c++98"}, {standard::cpp03, "c++03"}
	, {standard::cpp11, "c++11"}, {standard::cpp14, "c++14"}
	});

	JLM_ASSERT(map.find(std) != map.end());
	return map[std];
}

/* cmdline parser */

static bool
is_objfile(const jlm::filepath & file)
{
	return file.suffix() == "o";
}

static jlm::filepath
to_objfile(const jlm::filepath & f)
{
	return jlm::filepath(f.path() + f.base() + ".o");
}

static jlm::filepath
ToDependencyFile(const jlm::filepath & f)
{
	return jlm::filepath(f.path() + f.base() + ".d");
}

void
parse_cmdline(int argc, char ** argv, jlm::JhlsCommandLineOptions & options)
{
	using namespace llvm;

	/*
		FIXME: The command line parser setup is currently redone
		for every invocation of parse_cmdline. We should be able
		to do it only once and then reset the parser on every
		invocation of parse_cmdline.
	*/

	cl::TopLevelSubCommand->reset();

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

	cl::opt<std::string> optlvl(
	  "O"
	, cl::Prefix
	, cl::ValueOptional
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

	cl::list<std::string> flags(
	  "f"
	, cl::Prefix
	, cl::desc("Specify flags.")
	, cl::value_desc("flag"));

	cl::list<std::string> jlmhls(
	  "J"
	, cl::Prefix
	, cl::desc("jlm-hls optimization. Run 'jlm-hls -help' for viable options.")
	, cl::value_desc("jlmhls"));

	cl::opt<bool> verbose(
	  "v"
	, cl::ValueDisallowed
	, cl::desc("Show commands to run and use verbose output. (Affects only clang for now)"));

	cl::opt<bool> rdynamic(
	  "rdynamic"
	, cl::ValueDisallowed
	, cl::desc("rdynamic option passed to clang"));

	cl::opt<bool> suppress(
	  "w"
	, cl::ValueDisallowed
	, cl::desc("Suppress all warnings"));

	cl::opt<bool> pthread(
	  "pthread"
	, cl::ValueDisallowed
	, cl::desc("Support POSIX threads in generated code"));

	cl::opt<bool> MD(
	  "MD"
	, cl::ValueDisallowed
	, cl::desc("Write a depfile containing user and system headers"));

	cl::opt<std::string> MF(
	  "MF"
	, cl::desc("Write depfile output from -MD to <file>.")
	, cl::value_desc("file"));

	cl::opt<std::string> MT(
	  "MT"
	, cl::desc("Specify name of main file output in depfile.")
	, cl::value_desc("value"));

	cl::list<std::string> hls_function(
	  "hls-function"
	, cl::Prefix
	, cl::desc("function that should be accelerated")
	, cl::value_desc("regex"));

	cl::opt<bool> generate_firrtl(
	  "firrtl"
	, cl::ValueDisallowed
	, cl::desc("Generate firrtl"));

	cl::opt<bool> circt(
	  "circt"
	, cl::Prefix
	, cl::desc("Use CIRCT to generate FIRRTL"));

	cl::ParseCommandLineOptions(argc, argv);

	/* Process parsed options */

	static std::unordered_map<std::string, jlm::optlvl> Olvlmap({
	  {"0", optlvl::O0}, {"1", optlvl::O1}
	, {"2", optlvl::O2}, {"3", optlvl::O3}}
	);

	static std::unordered_map<std::string, standard> stdmap({
	  {"gnu89", standard::gnu89}, {"gnu99", standard::gnu99}
	, {"c89", standard::c89}, {"c90", standard::c99}
	, {"c99", standard::c99}, {"c11", standard::c11}
	, {"c++98", standard::cpp98}, {"c++03", standard::cpp03}
	, {"c++11", standard::cpp11}, {"c++14", standard::cpp14}
	});

	if (!optlvl.empty()) {
		auto olvl = Olvlmap.find(optlvl);
		if (olvl == Olvlmap.end()) {
			std::cerr << "Unknown optimization level.\n";
			exit(EXIT_FAILURE);
		}
		options.Olvl = olvl->second;
	}

	if (!std.empty()) {
		auto stdit = stdmap.find(std);
		if (stdit == stdmap.end()) {
			std::cerr << "Unknown language standard.\n";
			exit(EXIT_FAILURE);
		}
		options.std = stdit->second;
	}

	if (ifiles.empty()) {
		std::cerr << "jlc: no input files.\n";
		exit(EXIT_FAILURE);
	}

	if (ifiles.size() > 1 && no_linking && !ofilepath.empty()) {
		std::cerr << "jlc: cannot specify -o when generating multiple output files.\n";
		exit(EXIT_FAILURE);
	}

	if (!hls_function.empty()) {
		options.hls = true;
		options.hls_function_regex = hls_function.front();
	}

	if (hls_function.size() > 1) {
		std::cerr << "jlc-hls: more than one function regex specified\n";
		exit(EXIT_FAILURE);
	}

	options.libs = libs;
	options.macros = Dmacros;
	options.libpaths = libpaths;
	options.warnings = Wwarnings;
	options.includepaths = includepaths;
	options.only_print_commands = print_commands;
	options.generate_debug_information = generate_debug_information;
	options.flags = flags;
	options.jlmhls = jlmhls;
	options.verbose = verbose;
	options.rdynamic = rdynamic;
	options.suppress = suppress;
	options.pthread = pthread;
	options.MD = MD;
	options.generate_firrtl = generate_firrtl;
	options.circt = circt;

	for (const auto & ifile : ifiles) {
		if (is_objfile(ifile)) {
			/* FIXME: print a warning like clang if no_linking is true */
			options.compilations.push_back({
				ifile,
				jlm::filepath(""),
				ifile,
				"",
				false,
				false,
				false,
				true});

			continue;
		}

		options.compilations.push_back({
			ifile,
			MF.empty() ? ToDependencyFile(ifile) : jlm::filepath(MF),
			to_objfile(ifile),
			MT.empty() ? to_objfile(ifile).name() : MT,
			true,
			true,
			true,
			!no_linking});
	}

	if (!ofilepath.empty()) {
		if (no_linking) {
			JLM_ASSERT(options.compilations.size() == 1);
			options.compilations[0].set_ofile(ofilepath);
		} else {
			options.lnkofile = jlm::filepath(ofilepath);
		}
	}
}

}

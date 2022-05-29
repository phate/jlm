/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlc/cmdline.hpp>

#include <llvm/Support/CommandLine.h>

#include <iostream>
#include <unordered_map>

namespace jlm {

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
parse_cmdline(int argc, char ** argv, JlcCommandLineOptions & options)
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

	cl::list<std::string> jlmopts(
	  "J"
	, cl::Prefix
	, cl::desc("jlm-opt optimization. Run 'jlm-opt -help' for viable options.")
	, cl::value_desc("jlmopt"));

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

	cl::ParseCommandLineOptions(argc, argv);

	/* Process parsed options */

	static std::unordered_map<std::string, jlm::JlcCommandLineOptions::OptimizationLevel> Olvlmap({
	  {"0", JlcCommandLineOptions::OptimizationLevel::O0},
    {"1", JlcCommandLineOptions::OptimizationLevel::O1},
    {"2", JlcCommandLineOptions::OptimizationLevel::O2},
    {"3", JlcCommandLineOptions::OptimizationLevel::O3}
  });

	static std::unordered_map<std::string, jlm::JlcCommandLineOptions::LanguageStandard> stdmap({
	  {"gnu89", jlm::JlcCommandLineOptions::LanguageStandard::Gnu89},
    {"gnu99", jlm::JlcCommandLineOptions::LanguageStandard::Gnu99},
    {"c89",   jlm::JlcCommandLineOptions::LanguageStandard::C89},
    {"c90",   jlm::JlcCommandLineOptions::LanguageStandard::C99},
    {"c99",   jlm::JlcCommandLineOptions::LanguageStandard::C99},
    {"c11",   jlm::JlcCommandLineOptions::LanguageStandard::C11},
    {"c++98", jlm::JlcCommandLineOptions::LanguageStandard::Cpp98},
    {"c++03", jlm::JlcCommandLineOptions::LanguageStandard::Cpp03},
    {"c++11", jlm::JlcCommandLineOptions::LanguageStandard::Cpp11},
    {"c++14", jlm::JlcCommandLineOptions::LanguageStandard::Cpp14}
	});

	if (!optlvl.empty()) {
		auto olvl = Olvlmap.find(optlvl);
		if (olvl == Olvlmap.end()) {
			std::cerr << "Unknown optimization level.\n";
			exit(EXIT_FAILURE);
		}
		options.OptimizationLevel_ = olvl->second;
	}

	if (!std.empty()) {
		auto stdit = stdmap.find(std);
		if (stdit == stdmap.end()) {
			std::cerr << "Unknown language standard.\n";
			exit(EXIT_FAILURE);
		}
		options.LanguageStandard_ = stdit->second;
	}

	if (ifiles.empty()) {
		std::cerr << "jlc: no input files.\n";
		exit(EXIT_FAILURE);
	}

	if (ifiles.size() > 1 && no_linking && !ofilepath.empty()) {
		std::cerr << "jlc: cannot specify -o when generating multiple output files.\n";
		exit(EXIT_FAILURE);
	}

	options.Libraries_ = libs;
	options.MacroDefinitions_ = Dmacros;
	options.LibraryPaths_ = libpaths;
	options.Warnings_ = Wwarnings;
	options.IncludePaths_ = includepaths;
	options.OnlyPrintCommands_ = print_commands;
	options.GenerateDebugInformation_ = generate_debug_information;
	options.Flags_ = flags;
	options.JlmOptOptimizations_ = jlmopts;
	options.Verbose_ = verbose;
	options.Rdynamic_ = rdynamic;
	options.Suppress_ = suppress;
	options.UsePthreads_ = pthread;
	options.Md_ = MD;

	for (const auto & ifile : ifiles) {
		if (is_objfile(ifile)) {
			/* FIXME: print a warning like clang if no_linking is true */
			options.Compilations_.push_back({
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

		options.Compilations_.push_back({
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
			JLM_ASSERT(options.Compilations_.size() == 1);
      options.Compilations_[0].SetOutputFile(ofilepath);
		} else {
			options.OutputFile_ = jlm::filepath(ofilepath);
		}
	}
}

}

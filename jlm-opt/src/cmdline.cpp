/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm-opt/cmdline.hpp>

#include <llvm/Support/CommandLine.h>

namespace jlm {

void
parse_cmdline(int argc, char ** argv, jlm::cmdline_options & options)
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

	cl::opt<std::string> ifile(
	  cl::Positional
	, cl::desc("<input>"));

	cl::opt<std::string> ofile(
	  "o"
	, cl::desc("Write output to <file>")
	, cl::value_desc("file"));

	std::string desc("Write stats to <file>. Default is " + options.sd.file().path().to_str() + ".");
	cl::opt<std::string> sfile(
	  "s"
	, cl::desc(desc)
	, cl::value_desc("file"));

	cl::opt<bool> print_cfr_time(
	  "print-cfr-time"
	, cl::ValueDisallowed
	, cl::desc("Write CFR time to stats file."));

	cl::opt<bool> print_aggregation_time(
	  "print-aggregation-time"
	, cl::ValueDisallowed
	, cl::desc("Write aggregation time to stats file."));

	cl::opt<bool> print_annotation_time(
	  "print-annotation-time"
	, cl::ValueDisallowed
	, cl::desc("Write annotation time to stats file."));

	cl::opt<outputformat> format(
	  cl::values(
		  clEnumValN(outputformat::llvm, "llvm", "Output LLVM IR [default]")
		, clEnumValN(outputformat::xml, "xml", "Output XML"))
	, cl::desc("Select output format"));

	cl::list<jlm::optimization> optimizations(
		cl::values(
		  clEnumValN(jlm::optimization::cne, "cne", "Common node elimination")
		, clEnumValN(jlm::optimization::dne, "dne", "Dead node elimination")
		, clEnumValN(jlm::optimization::iln, "iln", "Function inlining")
		, clEnumValN(jlm::optimization::inv, "inv", "Invariant value reduction")
		, clEnumValN(jlm::optimization::psh, "psh", "Node push out")
		, clEnumValN(jlm::optimization::pll, "pll", "Node pull in")
		, clEnumValN(jlm::optimization::red, "red", "Node reductions")
		, clEnumValN(jlm::optimization::ivt, "ivt", "Theta-gamma inversion")
		, clEnumValN(jlm::optimization::url, "url", "Loop unrolling"))
	, cl::desc("Perform optimization"));

	cl::ParseCommandLineOptions(argc, argv);

	if (show_help) {
		cl::PrintHelpMessage();
		exit(EXIT_SUCCESS);
	}

	if (!ofile.empty())
		options.ofile = ofile;

	if (!sfile.empty())
		options.sd.set_file(sfile);

	options.ifile = ifile;
	options.format = format;
	options.optimizations = optimizations;
	options.sd.print_cfr_time = print_cfr_time;
	options.sd.print_annotation_time = print_annotation_time;
	options.sd.print_aggregation_time = print_aggregation_time;
}

}

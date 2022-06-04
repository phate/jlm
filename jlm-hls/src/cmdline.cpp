/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm-hls/cmdline.hpp>

#include <llvm/Support/CommandLine.h>

namespace jlm {

void
parse_cmdline(int argc, char ** argv, jlm::JlmHlsCommandLineOptions & options)
{
	using namespace llvm;

	/*
		FIXME: The command line parser setup is currently redone
		for every invocation of parse_cmdline. We should be able
		to do it only once and then reset the parser on every
		invocation of parse_cmdline.
	*/

	cl::TopLevelSubCommand->reset();

	cl::opt<std::string> inputFile(
	  cl::Positional
	, cl::desc("<input>"));

	cl::opt<std::string> outputFolder(
	  "o"
	, cl::desc("Write output to <folder>")
	, cl::value_desc("folder"));

	cl::opt<std::string> hlsFunction(
	  "hls-function"
	, cl::Prefix
	, cl::desc("Function that should be accelerated")
	, cl::value_desc("hls-function"));

	cl::opt<bool> extractHlsFunction(
	  "extract"
	, cl::Prefix
	, cl::desc("Extracts function specified by hls-function"));

	cl::opt<bool> useCirct(
	  "circt"
	, cl::Prefix
	, cl::desc("Use CIRCT to generate FIRRTL"));

  cl::opt<JlmHlsCommandLineOptions::OutputFormat> format(
    cl::values(
      clEnumValN(
        JlmHlsCommandLineOptions::OutputFormat::Firrtl,
        "fir",
        "Output FIRRTL [default]"),
      clEnumValN(
        JlmHlsCommandLineOptions::OutputFormat::Dot,
        "dot",
        "Output DOT graph")),
    cl::desc("Select output format"));

	cl::ParseCommandLineOptions(argc, argv);

	if (outputFolder.empty()) {
		throw jlm::error("jlm-hls no output directory provided, i.e, -o.\n");
	}

	if (extractHlsFunction && hlsFunction.empty()) {
		throw jlm::error("jlm-hls: --hls-function is not specified.\n         which is required for --extract\n");
	}

	options.InputFile_ = inputFile;
	options.HlsFunction_ = std::move(hlsFunction);
	options.OutputFolder_ = outputFolder;
	options.ExtractHlsFunction_ = extractHlsFunction;
	options.UseCirct_ = useCirct;
	options.OutputFormat_ = format;
}

} // jlm

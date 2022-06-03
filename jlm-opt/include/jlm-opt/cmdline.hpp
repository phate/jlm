/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLMOPT_CMDLINE_HPP
#define JLM_JLMOPT_CMDLINE_HPP

#include <jlm/util/file.hpp>
#include <jlm/util/Statistics.hpp>

#include <string>
#include <vector>

namespace jlm {

class optimization;

class JlmOptCommandLineOptions {
public:
  enum class OutputFormat {
    Llvm,
    Xml
  };

	JlmOptCommandLineOptions()
	: InputFile_("")
	, OutputFile_("")
	, OutputFormat_(OutputFormat::Llvm)
	{}

	filepath InputFile_;
	filepath OutputFile_;
	OutputFormat OutputFormat_;
	StatisticsDescriptor StatisticsDescriptor_;
	std::vector<optimization*> Optimizations_;
};

void
parse_cmdline(int argc, char ** argv, JlmOptCommandLineOptions & options);

}

#endif

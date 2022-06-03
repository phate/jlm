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

enum class outputformat {llvm, xml};

class JlmOptCommandLineOptions {
public:
	JlmOptCommandLineOptions()
	: ifile("")
	, ofile("")
	, format(outputformat::llvm)
	{}

	jlm::filepath ifile;
	jlm::filepath ofile;
	outputformat format;
	StatisticsDescriptor sd;
	std::vector<jlm::optimization*> optimizations;
};

void
parse_cmdline(int argc, char ** argv, JlmOptCommandLineOptions & options);

}

#endif

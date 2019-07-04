/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLMOPT_CMDLINE_HPP
#define JLM_JLMOPT_CMDLINE_HPP

#include <jlm/opt/optimization.hpp>
#include <jlm/util/file.hpp>
#include <jlm/util/stats.hpp>

#include <string>
#include <vector>

namespace jlm {

enum class outputformat {llvm, xml};

class cmdline_options {
public:
	cmdline_options()
	: ifile("")
	, ofile("")
	, format(outputformat::llvm)
	{}

	jlm::filepath ifile;
	jlm::filepath ofile;
	outputformat format;
	stats_descriptor sd;
	std::vector<jlm::optimization> optimizations;
};

void
parse_cmdline(int argc, char ** argv, cmdline_options & options);

}

#endif

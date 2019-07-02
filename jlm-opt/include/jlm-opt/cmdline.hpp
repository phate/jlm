/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLMOPT_CMDLINE_HPP
#define JLM_JLMOPT_CMDLINE_HPP

#include <jlm/opt/optimization.hpp>
#include <jlm/util/file.hpp>

#include <string>
#include <vector>

namespace jlm {

enum class outputformat {llvm, xml};

class cmdline_options {
public:
	cmdline_options()
	: ifile("")
	, format(outputformat::llvm)
	{}

	jlm::file ifile;
	outputformat format;
	std::vector<jlm::optimization> optimizations;
};

void
parse_cmdline(int argc, char ** argv, cmdline_options & options);

}

#endif

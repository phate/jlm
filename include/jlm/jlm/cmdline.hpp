/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <string>
#include <vector>

namespace jlm {

enum class optlvl {O0, O1, O2, O3};

std::string
to_str(const optlvl & ol);

class cmdline_options {
public:
	cmdline_options()
	: only_print_commands(false)
	, generate_debug_information(false)
	, Olvl(optlvl::O0)
	, ofilepath("a.out")
	{}

	bool only_print_commands;
	bool generate_debug_information;

	optlvl Olvl;
	std::string ofilepath;
	std::vector<std::string> libs;
	std::vector<std::string> libpaths;
	std::vector<std::string> ifilepaths;
	std::vector<std::string> includepaths;
};

void
parse_cmdline(int argc, char ** argv, cmdline_options & options);

}

/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <string>
#include <vector>

namespace jlm {

class cmdline_options {
public:
	cmdline_options()
	: only_print_commands(false)
	, ofilepath("a.out")
	{}

	bool only_print_commands;

	std::string ofilepath;
	std::vector<std::string> libpaths;
	std::vector<std::string> ifilepaths;
	std::vector<std::string> includepaths;
};

void
parse_cmdline(int argc, char ** argv, cmdline_options & options);

}

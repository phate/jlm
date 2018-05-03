/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <string>
#include <vector>

namespace jlm {

class cmdflags {
public:
	cmdflags()
	: only_print_commands(false)
	, ofilepath("a.out")
	{}

	bool only_print_commands;

	std::string ofilepath;
	std::vector<std::string> ifilepaths;
};

void
parse_cmdline(int argc, char ** argv, cmdflags & flags);

}

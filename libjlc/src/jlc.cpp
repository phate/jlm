/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlc/cmdline.hpp>
#include <jlc/command.hpp>

#include <iostream>

int
main(int argc, char ** argv)
{
	jlm::JlcCommandLineOptions commandLineOptions;
	parse_cmdline(argc, argv, commandLineOptions);

	auto pgraph = generate_commands(commandLineOptions);
  pgraph->Run();

	return 0;
}

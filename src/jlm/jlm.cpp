/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/jlm/cmdline.hpp>
#include <jlm/jlm/command.hpp>

#include <iostream>

int
main(int argc, char ** argv)
{
	jlm::cmdline_options options;
	parse_cmdline(argc, argv, options);

	auto cmds = generate_commands(options);

	if (options.only_print_commands) {
		for (const auto & cmd : cmds)
			std::cout << cmd->to_str() << "\n";
		return 0;
	}

	for (const auto & cmd : cmds)
		cmd->execute();

	return 0;
}

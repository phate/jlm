/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/jlc/cmdline.hpp>
#include <jlm/jlc/command.hpp>

#include <iostream>

int
main(int argc, char ** argv)
{
	jlm::cmdline_options options;
	parse_cmdline(argc, argv, options);

	auto cmds = generate_commands(options);
	for (const auto & cmd : cmds)
		cmd->execute();

	return 0;
}

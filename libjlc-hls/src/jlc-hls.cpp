/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlc-hls/cmdline.hpp>
#include <jlc-hls/command.hpp>

#include <iostream>

int
main(int argc, char ** argv)
{
	jlm::cmdline_options options;
	parse_cmdline(argc, argv, options);

	auto pgraph = generate_commands(options);
	pgraph->run();

	return 0;
}

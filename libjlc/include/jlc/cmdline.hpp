/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLC_CMDLINE_HPP
#define JLM_JLC_CMDLINE_HPP

#include <jlm/tooling/CommandLine.hpp>
#include <jlm/util/file.hpp>

#include <string>
#include <vector>

namespace jlm {

void
parse_cmdline(int argc, char ** argv, JlcCommandLineOptions & options);

}

#endif

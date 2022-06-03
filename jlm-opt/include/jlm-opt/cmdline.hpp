/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLMOPT_CMDLINE_HPP
#define JLM_JLMOPT_CMDLINE_HPP

#include <jlm/tooling/CommandLine.hpp>

namespace jlm {

void
parse_cmdline(int argc, char ** argv, JlmOptCommandLineOptions & options);

}

#endif

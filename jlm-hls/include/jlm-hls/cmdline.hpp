/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLMHLS_CMDLINE_HPP
#define JLM_JLMHLS_CMDLINE_HPP

#include <jlm/tooling/CommandLine.hpp>
#include <jlm/util/file.hpp>

namespace jlm {

void
parse_cmdline(int argc, char ** argv, JlmHlsCommandLineOptions & options);

}

#endif

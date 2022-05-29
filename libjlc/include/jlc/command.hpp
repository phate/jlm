/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_JLC_COMMAND_HPP
#define JLM_JLC_COMMAND_HPP

#include <jlc/cmdline.hpp>
#include <jlm/tooling/Command.hpp>
#include <jlm/tooling/CommandGraph.hpp>

#include <memory>
#include <string>
#include <vector>

namespace jlm {

std::unique_ptr<CommandGraph>
generate_commands(const JlcCommandLineOptions & commandLineOptions);

}

#endif

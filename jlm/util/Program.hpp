/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_PROGRAM_HPP
#define JLM_UTIL_PROGRAM_HPP

#include <filesystem>
#include <optional>
#include <vector>

namespace jlm::util
{

/**
 * Executes a program given by its path \p programPath and its arguments \p programArguments.
 *
 * @param programName The name of the program.
 * @param programArguments The arguments for the program.
 * @return The return code of the executed program, or EXIT_FAILURE if the program could not be
 * executed.
 */
int
executeProgramAndWait(
    const std::string & programName,
    const std::vector<std::string> & programArguments);

/**
 * @return The name of a dot viewer.
 */
std::string
getDotViewer();

}

#endif

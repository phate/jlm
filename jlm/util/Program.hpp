/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_PROGRAM_HPP
#define JLM_UTIL_PROGRAM_HPP

#include <filesystem>
#include <vector>

namespace jlm::util
{

/**
 * Tries to find the path of executable \p programName
 *
 * @param programName The name of the executable
 * @return The path to the executable if found, otherwise an empty path.
 */
std::filesystem::path
tryFindExecutablePath(std::string_view programName);

/**
 * Executes a program given by its path \p programPath and its arguments \p programArguments.
 *
 * @param programPath The path to the program.
 * @param programArguments The arguments for the program.
 * @return EXIT_FAILURE if the program could not be executed, otherwise the return code of the executed program.
 */
int
executeProgramAndWait(
    const std::filesystem::path & programPath,
    const std::vector<std::string> & programArguments);

/**
 * Tries to get the path to a dot viewer.
 *
 * @return The path to the executable if found, otherwise an empty path.
 */
std::filesystem::path
tryGetDotViewer();

}

#endif

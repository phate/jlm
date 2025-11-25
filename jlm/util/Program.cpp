/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/common.hpp>
#include <jlm/util/Program.hpp>
#include <jlm/util/strfmt.hpp>

namespace jlm::util
{

std::filesystem::path
tryFindExecutablePath(const std::string_view programName)
{
  const auto command = strfmt("which ", programName);
  FILE * pipe = popen(command.c_str(), "r");
  if (!pipe)
  {
    return std::filesystem::path();
  }

  char buffer[128];
  std::string result;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
  {
    result += buffer;
  }

  pclose(pipe);

  // Remove trailing newline character
  if (!result.empty() && result.back() == '\n')
  {
    result.pop_back();
  }

  return std::filesystem::path(result);
}

int
executeProgramAndWait(
    const std::filesystem::path & programPath,
    const std::vector<std::string> & programArguments)
{
  JLM_ASSERT(!programPath.empty());
  JLM_ASSERT(std::filesystem::is_regular_file(std::filesystem::status(programPath)));

  std::string command = programPath.string();
  for (auto & argument : programArguments)
  {
    command += strfmt(" ", argument);
  }

  return system(command.c_str());
}

std::filesystem::path
tryGetDotViewer()
{
  return tryFindExecutablePath("xdot");
}

}

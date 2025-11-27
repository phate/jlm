/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/common.hpp>
#include <jlm/util/Program.hpp>
#include <jlm/util/strfmt.hpp>

#include <spawn.h>
#include <sys/wait.h>

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
  std::vector<char *> args(programArguments.size() + 2, nullptr);
  args[0] = const_cast<char *>(programPath.filename().c_str());
  for (const auto & argument : programArguments)
  {
    args.push_back(const_cast<char *>(argument.c_str()));
  }

  pid_t pid;
  int status = posix_spawn(&pid, programPath.string().c_str(), nullptr, nullptr, args.data(), {});
  if (status != 0)
  {
    return EXIT_FAILURE;
  }

  waitpid(pid, &status, 0);
  return EXIT_SUCCESS;
}

std::filesystem::path
tryGetDotViewer()
{
  return tryFindExecutablePath("xdot");
}

}

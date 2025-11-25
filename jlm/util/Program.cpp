/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/common.hpp>
#include <jlm/util/Program.hpp>
#include <jlm/util/strfmt.hpp>

#include <sys/wait.h>
#include <unistd.h>

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
  const pid_t pid = fork();
  if (pid == -1)
  {
    return EXIT_FAILURE;
  }

  if (pid == 0)
  {
    std::vector<char *> arguments(programArguments.size() + 1, nullptr);
    for (const auto & argument : programArguments)
    {
      arguments.push_back(const_cast<char *>(argument.c_str()));
    }

    execv(programPath.string().data(), arguments.data());
    exit(EXIT_FAILURE);
  }

  int status = EXIT_FAILURE;
  waitpid(pid, &status, 0);

  return WEXITSTATUS(status);
}

std::filesystem::path
tryGetDotViewer()
{
  return tryFindExecutablePath("xdot");
}

}

/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/common.hpp>
#include <jlm/util/Program.hpp>

#include <spawn.h>
#include <sys/wait.h>

extern char ** environ;

namespace jlm::util
{

int
executeProgramAndWait(
    const std::string & programName,
    const std::vector<std::string> & programArguments)
{
  std::vector<char *> arguments;
  arguments.push_back(const_cast<char *>(programName.c_str()));
  for (const auto & argument : programArguments)
  {
    arguments.push_back(const_cast<char *>(argument.c_str()));
  }
  arguments.push_back(nullptr);

  pid_t pid = -1;
  int status = posix_spawnp(&pid, programName.c_str(), nullptr, nullptr, arguments.data(), environ);
  if (status != 0)
  {
    return EXIT_FAILURE;
  }

  waitpid(pid, &status, 0);
  return EXIT_SUCCESS;
}

std::string
getDotViewer()
{
  return "xdot";
}

}

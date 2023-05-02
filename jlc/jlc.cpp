/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/tooling/CommandGraphGenerator.hpp>
#include <jlm/llvm/tooling/CommandLine.hpp>

int
main(int argc, char ** argv)
{
  jlm::JlcCommandLineParser commandLineParser;
  auto & commandLineOptions = commandLineParser.ParseCommandLineArguments(argc, argv);

  auto commandGraph = jlm::JlcCommandGraphGenerator::Generate(commandLineOptions);
  commandGraph->Run();

  return 0;
}

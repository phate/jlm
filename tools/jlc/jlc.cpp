/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/tooling/CommandGraphGenerator.hpp>
#include <jlm/tooling/CommandLine.hpp>

int
main(int argc, char ** argv)
{
  using namespace jlm::tooling;

  JlcCommandLineParser commandLineParser;
  const JlcCommandLineOptions * commandLineOptions = nullptr;
  try
  {
    commandLineOptions = &commandLineParser.ParseCommandLineArguments(argc, argv);
  }
  catch (const CommandLineParser::Exception & e)
  {
    std::cerr << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  auto commandGraph = JlcCommandGraphGenerator::Generate(*commandLineOptions);
  commandGraph->Run();

  return 0;
}

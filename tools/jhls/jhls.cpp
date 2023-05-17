/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/tooling/CommandLine.hpp>
#include <jlm/tooling/CommandGraphGenerator.hpp>

int
main(int argc, char ** argv)
{
  using namespace jlm::tooling;

  auto & commandLineOptions = JhlsCommandLineParser::Parse(argc, argv);

  auto commandGraph = JhlsCommandGraphGenerator::Generate(commandLineOptions);
  commandGraph->Run();

  return 0;
}

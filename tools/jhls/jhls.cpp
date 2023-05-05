/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/tooling/CommandLine.hpp>
#include <jlm/tooling/CommandGraphGenerator.hpp>

int
main(int argc, char ** argv)
{
  auto & commandLineOptions = jlm::JhlsCommandLineParser::Parse(argc, argv);

  auto commandGraph = jlm::JhlsCommandGraphGenerator::Generate(commandLineOptions);
  commandGraph->Run();

  return 0;
}

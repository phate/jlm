/*
 * Copyright 2022 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/tooling/CommandGraphGenerator.hpp>
#include <jlm/tooling/CommandLine.hpp>

int
main(int argc, char ** argv)
{
#ifndef CIRCT
  ::llvm::outs() << "jhls has not been compiled with the CIRCT backend enabled.\n";
  ::llvm::outs() << "Recompile jlm with -DCIRCT=1 if you want to use jhls.\n";
  exit(0);
#endif

  using namespace jlm::tooling;

  auto & commandLineOptions = JhlsCommandLineParser::Parse(argc, argv);

  auto commandGraph = JhlsCommandGraphGenerator::Generate(commandLineOptions);
  commandGraph->Run();

  return 0;
}

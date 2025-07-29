/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/tooling/Command.hpp>

int
main(int argc, char ** argv)
{
  auto & commandLineOptions = jlm::tooling::JlmOptCommandLineParser::Parse(argc, argv);

  try
  {
    jlm::tooling::JlmOptCommand command(argv[0], commandLineOptions);
    command.Run();
  }
  catch (jlm::util::Error & e)
  {
    std::cerr << e.what();
    exit(EXIT_FAILURE);
  }

  return 0;
}

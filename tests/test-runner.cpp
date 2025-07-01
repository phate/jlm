/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

int
main(const int argc, char ** argv)
{
  if (argc < 2)
  {
    jlm::tests::RunAllUnitTests();
  }
  else
  {
    for (int n = 1; n < argc; ++n)
    {
      jlm::tests::run_unit_test(argv[n]);
    }
  }

  return 0;
}

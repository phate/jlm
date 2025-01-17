/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

int
main(int argc, char ** argv)
{
  if (argc < 2)
  {
    return jlm::tests::RunAllUnitTests();
  }
  else
  {
    int fail = 0;
    for (int n = 1; n < argc; ++n)
    {
      if (jlm::tests::run_unit_test(argv[n]))
      {
        fail = 1;
      }
    }
    return fail;
  }
}

/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <cassert>
#include <iostream>
#include <map>
#include <memory>

namespace jlm::tests
{

class unit_test
{
public:
  unit_test(int (*v)())
      : verify(v)
  {}

  int (*verify)();
};

using unit_test_map_t = std::map<std::string, std::unique_ptr<unit_test>>;

static unit_test_map_t &
GetUnitTestMap()
{
  static unit_test_map_t unit_test_map;
  return unit_test_map;
}

void
register_unit_test(const std::string & name, int (*verify)())
{
  assert(GetUnitTestMap().find(name) == GetUnitTestMap().end());
  GetUnitTestMap().insert(std::make_pair(name, std::make_unique<unit_test>(verify)));
}

int
run_unit_test(const std::string & name)
{
  assert(GetUnitTestMap().find(name) != GetUnitTestMap().end());
  return GetUnitTestMap()[name]->verify();
}

int
RunAllUnitTests()
{
  int fail = 0;
  for (const auto & named_test : GetUnitTestMap())
  {
    std::cerr << named_test.first << std::endl;
    if (named_test.second->verify())
    {
      fail = 1;
    }
  }
  return fail;
}

}

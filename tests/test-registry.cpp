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

class UnitTest
{
public:
  explicit UnitTest(void (*v)())
      : verify(v)
  {}

  void (*verify)();
};

using unit_test_map_t = std::map<std::string, std::unique_ptr<UnitTest>>;

static unit_test_map_t &
GetUnitTestMap()
{
  static unit_test_map_t unit_test_map;
  return unit_test_map;
}

void
register_unit_test(const std::string & name, void (*verify)())
{
  assert(GetUnitTestMap().find(name) == GetUnitTestMap().end());
  GetUnitTestMap().insert(std::make_pair(name, std::make_unique<UnitTest>(verify)));
}

void
run_unit_test(const std::string & name)
{
  assert(GetUnitTestMap().find(name) != GetUnitTestMap().end());
  return GetUnitTestMap()[name]->verify();
}

void
RunAllUnitTests()
{
  for (const auto & [testName, unitTest] : GetUnitTestMap())
  {
    std::cerr << testName << std::endl;
    unitTest->verify();
  }
}

}

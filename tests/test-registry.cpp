/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <assert.h>
#include <memory>
#include <unordered_map>

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

static std::unordered_map<std::string, std::unique_ptr<unit_test>> &
GetUnitTestMap()
{
  static std::unordered_map<std::string, std::unique_ptr<unit_test>> unit_test_map;
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

}

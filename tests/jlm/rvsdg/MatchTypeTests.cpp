/*
 * Copyright 2025 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <cassert>

#include "jlm/rvsdg/MatchType.hpp"
#include "test-registry.hpp"

namespace
{

class Base
{
public:
  virtual ~Base() = default;
};

class X final : public Base
{
public:
  ~X() override = default;
};

class Y final : public Base
{
public:
  ~Y() override = default;
};

class Z final : public Base
{
public:
  ~Z() override = default;
};

class U final : public Base
{
public:
  ~U() override = default;
};

int
Discriminate1(const Base & obj)
{
  int result = -1;
  jlm::rvsdg::MatchType(
      obj,
      [&result](const X &)
      {
        result = 0;
      },
      [&result](const Y &)
      {
        result = 1;
      },
      [&result](const Z &)
      {
        result = 2;
      });
  return result;
}

int
Discriminate2(const Base & obj)
{
  int result = 42;
  jlm::rvsdg::MatchTypeWithDefault(
      obj,
      [&result](const X &)
      {
        result = 0;
      },
      [&result](const Y &)
      {
        result = 1;
      },
      [&result](const Z &)
      {
        result = 2;
      },
      [&result]()
      {
        result = -1;
      });
  return result;
}

int
Discriminate3(const Base & obj)
{
  return jlm::rvsdg::MatchTypeWithDefault(
      obj,
      [](const X &)
      {
        return 0;
      },
      [](const Y &)
      {
        return 1;
      },
      [](const Z &)
      {
        return 2;
      },
      []()
      {
        return -1;
      });
}

int
Discriminate4(const Base & obj)
{
  return jlm::rvsdg::MatchTypeOrFail(
      obj,
      [](const X &)
      {
        return 0;
      },
      [](const Y &)
      {
        return 1;
      },
      [](const Z &)
      {
        return 2;
      });
}

int
HandleX(const X &)
{
  return 0;
}

int
HandleY(const Y &)
{
  return 1;
}

int
Default()
{
  return 2;
}

int
Discriminate5(const Base & obj)
{
  return jlm::rvsdg::MatchTypeWithDefault(obj, HandleX, HandleY, Default);
}

}

static void
TestBasicTypeMatch()
{
  assert(Discriminate1(X()) == 0);
  assert(Discriminate1(Y()) == 1);
  assert(Discriminate1(Z()) == 2);
  assert(Discriminate1(U()) == -1);

  assert(Discriminate2(X()) == 0);
  assert(Discriminate2(Y()) == 1);
  assert(Discriminate2(Z()) == 2);
  assert(Discriminate2(U()) == -1);

  assert(Discriminate3(X()) == 0);
  assert(Discriminate3(Y()) == 1);
  assert(Discriminate3(Z()) == 2);
  assert(Discriminate3(U()) == -1);

  assert(Discriminate4(X()) == 0);
  assert(Discriminate4(Y()) == 1);
  assert(Discriminate4(Z()) == 2);
  try
  {
    Discriminate4(U());
    assert(false);
  }
  catch (const std::logic_error &)
  {
    assert(true);
  }

  assert(Discriminate5(X()) == 0);
  assert(Discriminate5(Y()) == 1);
  assert(Discriminate5(Z()) == 2);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/MatchTypeTests-TestBasicTypeMatch", TestBasicTypeMatch)

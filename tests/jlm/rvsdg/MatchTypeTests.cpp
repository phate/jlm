/*
 * Copyright 2025 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <cassert>

#include <gtest/gtest.h>

#include <jlm/rvsdg/MatchType.hpp>

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

TEST(MatchTypeTests, TestBasicTypeMatch)
{
  EXPECT_EQ(Discriminate1(X()), 0);
  EXPECT_EQ(Discriminate1(Y()), 1);
  EXPECT_EQ(Discriminate1(Z()), 2);
  EXPECT_EQ(Discriminate1(U()), -1);

  EXPECT_EQ(Discriminate2(X()), 0);
  EXPECT_EQ(Discriminate2(Y()), 1);
  EXPECT_EQ(Discriminate2(Z()), 2);
  EXPECT_EQ(Discriminate2(U()), -1);

  EXPECT_EQ(Discriminate3(X()), 0);
  EXPECT_EQ(Discriminate3(Y()), 1);
  EXPECT_EQ(Discriminate3(Z()), 2);
  EXPECT_EQ(Discriminate3(U()), -1);

  EXPECT_EQ(Discriminate4(X()), 0);
  EXPECT_EQ(Discriminate4(Y()), 1);
  EXPECT_EQ(Discriminate4(Z()), 2);
  EXPECT_THROW(Discriminate4(U()), std::logic_error);

  EXPECT_EQ(Discriminate5(X()), 0);
  EXPECT_EQ(Discriminate5(Y()), 1);
  EXPECT_EQ(Discriminate5(Z()), 2);
}

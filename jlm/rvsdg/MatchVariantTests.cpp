/*
 * Copyright 2026 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/MatchVariant.hpp>

namespace
{

using TestVariant = std::variant<int, std::string>;

std::string
Discriminate(const TestVariant & v)
{
  return jlm::rvsdg::MatchVariant(
      v,
      [](const int & n)
      {
        return "int:" + std::to_string(n);
      },
      [](const std::string & s)
      {
        return "str:" + s;
      });
}

}

TEST(MatchVariantTests, TestVariants)
{
  EXPECT_EQ(Discriminate(TestVariant(42)), "int:42");
  EXPECT_EQ(Discriminate(TestVariant("foo")), "str:foo");
}

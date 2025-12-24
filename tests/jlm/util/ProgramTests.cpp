/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/util/Program.hpp>

#include <cassert>

TEST(ProgramTests, testExecuteProgramAndWait)
{
  using namespace jlm::util;

  {
    const auto status =
        executeProgramAndWait("ls", { std::filesystem::temp_directory_path().string() });
    EXPECT_EQ(status, EXIT_SUCCESS);
  }
}

/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/Program.hpp>

#include <cassert>

static void
testExecuteProgramAndWait()
{
  using namespace jlm::util;

  {
    const auto status =
        executeProgramAndWait("ls", { std::filesystem::temp_directory_path().string() });
    assert(status == EXIT_SUCCESS);
  }
}

JLM_UNIT_TEST_REGISTER("jlm/util/ProgramTests-testExecuteProgramAndWait", testExecuteProgramAndWait)

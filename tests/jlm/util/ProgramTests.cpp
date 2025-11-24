/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/Program.hpp>

#include <cassert>

static void
testTryFindExecutablePath()
{
  using namespace jlm::util;

  // Test unknown executable
  {
    // This assumes that this executable does not exist
    const auto path = tryFindExecutablePath("xyz123");
    assert(path.empty());
  }

  // Test known executable
  {
    const auto path = tryFindExecutablePath("ls");
    assert(!path.empty());
  }
}

JLM_UNIT_TEST_REGISTER("jlm/util/ProgramTests-testTryFindExecutablePath", testTryFindExecutablePath)

static void
testExecuteProgramAndWait()
{
  using namespace jlm::util;

  const auto path = tryFindExecutablePath("ls");
  assert(!path.empty());

  const auto returnValue =
      executeProgramAndWait(path, { std::filesystem::temp_directory_path().string() });
  assert(returnValue == 0);
}

JLM_UNIT_TEST_REGISTER("jlm/util/ProgramTests-testExecuteProgramAndWait", testExecuteProgramAndWait)

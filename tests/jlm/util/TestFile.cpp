/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/file.hpp>

#include <cassert>

// FIXME: This was shamelessly copied from util/file.hpp.
// We should introduce our own string wrapper where we can have such methods.
static std::string
CreateRandomString(std::size_t length)
{
  const std::string characterSet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::uniform_int_distribution<> distribution(0, characterSet.size() - 1);

  std::string result;
  for (std::size_t i = 0; i < length; ++i)
  {
    result += characterSet[distribution(generator)];
  }

  return result;
}

static void
TestFilePathMethods()
{
  jlm::util::filepath f("/tmp/archive.tar.gz");

  assert(f.name() == "archive.tar.gz");
  assert(f.base() == "archive");
  assert(f.suffix() == "gz");
  assert(f.complete_suffix() == "tar.gz");
  assert(f.path() == "/tmp/");
}

static void
TestCreateUniqueFileName()
{
  // Arrange
  auto randomString = CreateRandomString(6);
  auto tmpDirectory = std::filesystem::temp_directory_path().string() + "/" + randomString;

  // Act
  auto filePath = jlm::util::filepath::CreateUniqueFileName(tmpDirectory, "myPrefix", "mySuffix");

  // Assert
  assert(filePath.path() == (tmpDirectory + "/"));
}

static int
TestFile()
{
  TestFilePathMethods();
  TestCreateUniqueFileName();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestFile", TestFile)

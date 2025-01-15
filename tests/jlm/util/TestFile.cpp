/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/file.hpp>

#include <cassert>
#include <vector>

static int
TestFilePathMethods()
{
  const jlm::util::filepath f("/tmp/archive.tar.gz");

  assert(f.to_str() == "/tmp/archive.tar.gz");
  assert(f.name() == "archive.tar.gz");
  assert(f.base() == "archive");
  assert(f.suffix() == "gz");
  assert(f.complete_suffix() == "tar.gz");
  assert(f.path() == "/tmp/");

  std::vector<std::pair<std::string, std::string>> pathPairs = { { "/tmp/jlm/", "/tmp/" },
                                                                 { "/tmp/", "/" },
                                                                 { "d/d2/file.txt", "d/d2/" },
                                                                 { "test.txt", "" },
                                                                 { "./test2.txt", "./" },
                                                                 { "/", "/" },
                                                                 { ".", "" },
                                                                 { "", "" } };
  for (const auto & [fullPath, path] : pathPairs)
  {
    assert(jlm::util::filepath(fullPath).path() == path);
  }

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestFile-TestFilePathMethods", TestFilePathMethods)

static int
TestCreateDirectory()
{
  const auto path = std::filesystem::temp_directory_path() / "jlm-test-create-dir";
  const jlm::util::filepath filepath(path);

  // Remove the directory if it survived from a previous test
  if (filepath.Exists())
    std::filesystem::remove(path);
  assert(!filepath.Exists());

  // Act
  filepath.CreateDirectory();

  // Assert that the directory now exists
  assert(filepath.Exists() && filepath.IsDirectory());

  // Try creating a directory that already exists, should be no issue
  filepath.CreateDirectory();

  // Try creating a directory in a location that does not exist
  try
  {
    jlm::util::filepath noSuchParent("/non-existant/test-dir");
    filepath.CreateDirectory();
    assert(false);
  }
  catch (...)
  {}

  // Cleanup
  std::filesystem::remove(path);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestFile-TestCreateDirectory", TestCreateDirectory)

static int
TestFilepathJoin()
{
  const jlm::util::filepath path1("tmp");
  const jlm::util::filepath path2("a/b/");
  const jlm::util::filepath path3("/c/d");

  assert(path1.join(path2).to_str() == "tmp/a/b/");
  assert(path2.join(path1).to_str() == "a/b/tmp");

  assert(path1.join(path3).to_str() == "/c/d");
  assert(path3.join(path1).to_str() == "/c/d/tmp");

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestFile-TestFilepathJoin", TestFilepathJoin)

static int
TestCreateUniqueFileName()
{
  // Arrange
  auto randomString = jlm::util::CreateRandomAlphanumericString(6);
  auto tmpDirectory = std::filesystem::temp_directory_path().string() + "/" + randomString;

  // Act
  auto filePath = jlm::util::filepath::CreateUniqueFileName(tmpDirectory, "myPrefix", "mySuffix");

  // Assert
  assert(filePath.path() == (tmpDirectory + "/"));

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestFile-TestCreateUniqueFileName", TestCreateUniqueFileName)

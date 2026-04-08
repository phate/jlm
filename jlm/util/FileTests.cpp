/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/util/file.hpp>

#include <vector>

TEST(FileTests, TestFilePathMethods)
{
  const jlm::util::FilePath f("/tmp/archive.tar.gz");

  EXPECT_EQ(f.to_str(), "/tmp/archive.tar.gz");
  EXPECT_EQ(f.name(), "archive.tar.gz");
  EXPECT_EQ(f.base(), "archive");
  EXPECT_EQ(f.suffix(), "gz");
  EXPECT_EQ(f.complete_suffix(), "tar.gz");
  EXPECT_EQ(f.Dirname(), "/tmp");

  std::vector<std::pair<std::string, std::string>> pathPairs = { { "/tmp/jlm/", "/tmp" },
                                                                 { "/tmp/jlm", "/tmp" },
                                                                 { "/tmp/", "/" },
                                                                 { "/tmp", "/" },
                                                                 { "d/d2/file.txt", "d/d2" },
                                                                 { "test.txt", "." },
                                                                 { "./test2.txt", "." },
                                                                 { "a/..", "a" },
                                                                 { "/", "/" },
                                                                 { ".", "." },
                                                                 { "", "." } };
  for (const auto & [fullPath, path] : pathPairs)
  {
    const auto result = jlm::util::FilePath(fullPath).Dirname();
    EXPECT_EQ(result, path);
  }
}

TEST(FileTests, TestCreateDirectory)
{
  const auto filePath = jlm::util::FilePath::TempDirectoryPath().Join("jlm-test-create-dir");

  // Remove the directory if it survived from a previous test
  if (filePath.Exists())
    std::filesystem::remove(filePath.to_str());
  EXPECT_FALSE(filePath.Exists());

  // Act
  filePath.CreateDirectory();

  // Assert that the directory now exists
  EXPECT_TRUE(filePath.Exists() && filePath.IsDirectory());

  // Try creating a directory that already exists, should be no issue
  filePath.CreateDirectory();

  // Try creating a directory in a location that does not exist
  jlm::util::FilePath noSuchParent("/non-existant/test-dir");
  EXPECT_THROW(noSuchParent.CreateDirectory(), jlm::util::Error);

  // Cleanup
  std::filesystem::remove(filePath.to_str());
}

TEST(FileTests, TestFilepathJoin)
{
  const jlm::util::FilePath path1("tmp");
  const jlm::util::FilePath path2("a/b/");
  const jlm::util::FilePath path3("/c/d");
  const jlm::util::FilePath path4(".");
  const jlm::util::FilePath emptyPath("");

  EXPECT_EQ(path1.Join(path2).to_str(), "tmp/a/b/");
  EXPECT_EQ(path2.Join(path1).to_str(), "a/b/tmp");

  EXPECT_EQ(path1.Join(path3).to_str(), "/c/d");
  EXPECT_EQ(path3.Join(path1).to_str(), "/c/d/tmp");

  EXPECT_EQ(path4.Join(path1).to_str(), "tmp");
  EXPECT_EQ(path4.Join(path2).to_str(), "a/b/");
  EXPECT_EQ(path1.Join(path4).to_str(), "tmp/.");
  EXPECT_EQ(path2.Join(path4).to_str(), "a/b/.");

  EXPECT_EQ(emptyPath.Join(path1).to_str(), "tmp");
}

TEST(FileTests, TestCreateUniqueFileName)
{
  // Arrange
  auto randomString = jlm::util::CreateRandomAlphanumericString(6);
  auto tmpDirectory = jlm::util::FilePath::TempDirectoryPath().Join(randomString);

  // Act
  auto filePath = jlm::util::FilePath::createUniqueFileName(tmpDirectory, "myPrefix", "mySuffix");

  // Assert
  EXPECT_EQ(filePath.Dirname(), tmpDirectory.to_str());
}

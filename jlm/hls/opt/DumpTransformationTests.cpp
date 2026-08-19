/*
 * Copyright 2026 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <filesystem>
#include <regex>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include <jlm/hls/opt/DumpTransformation.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/util/strfmt.hpp>

namespace fs = std::filesystem;

namespace jlm::hls
{
using namespace jlm::llvm;
using namespace jlm::rvsdg;

TEST(DumpTransformationTests, DisabledDoesNothing)
{
  // Arrange — minimal module
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & rvsdg = rvsdgModule.Rvsdg();

  const auto functionType = FunctionType::Create({}, {});
  const auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  lambdaNode->finalize({});

  // Act — create disabled instance and run
  DumpTransformation dump(DumpTransformation::Disabled);
  jlm::util::StatisticsCollector statisticsCollector;

  EXPECT_NO_THROW(dump.Run(rvsdgModule, statisticsCollector));

  // Assert — no files created, Run() returns immediately
}

TEST(DumpTransformationTests, EnabledProducesOutputFiles)
{
  // Arrange — create a unique temp directory and pass it via the StatisticsCollector
  // so files don't pollute the project directory.
  auto tmpDirName = "jlm-dump-test-" + jlm::util::CreateRandomAlphanumericString(6);
  auto tmpDir = fs::temp_directory_path() / tmpDirName;
  fs::create_directories(tmpDir);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & rvsdg = rvsdgModule.Rvsdg();

  const auto functionType = FunctionType::Create({}, {});
  const auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  lambdaNode->finalize({});

  // Act — create enabled instance with all formats, run it
  DumpTransformation::Config config;
  config.formats = {
    DumpTransformation::OutputFormat::Dot,           DumpTransformation::OutputFormat::HlsDot,
    DumpTransformation::OutputFormat::StructuralDot, DumpTransformation::OutputFormat::Json,
    DumpTransformation::OutputFormat::Ascii,         DumpTransformation::OutputFormat::Tree,
    DumpTransformation::OutputFormat::JsonTree,
  };

  DumpTransformation dump(std::move(config));
  dump.setLabel("test-pass");
  dump.setPassNumber(42);

  jlm::util::StatisticsCollector statisticsCollector(jlm::util::StatisticsCollectorSettings(
      {},
      std::optional<jlm::util::FilePath>(tmpDir),
      "test-module"));

  EXPECT_NO_THROW(dump.Run(rvsdgModule, statisticsCollector));

  // Assert — all 7 files exist in the temp directory with correct naming pattern.
  // The unique string is randomly generated, so we use a regex to validate the format.
  std::set<std::string> expectedExtensions = {
    "llvm.dot", "hls.dot", "structural.dot", "json", "txt", "tree.txt", "tree.json",
  };

  auto entries = fs::directory_iterator(tmpDir);
  std::set<std::string> foundExtensions;
  for (const auto & entry : entries)
  {
    if (entry.is_regular_file())
    {
      auto fname = entry.path().filename();
      // Match the pattern: [modulename]-[unique]-NNN-label.ext
      // Accept any prefix up to "-test-pass.", then capture the extension
      std::string str = fname;
      std::regex pattern("-042-test-pass\\.(.*)");
      std::smatch match;
      if (std::regex_search(str, match, pattern))
      {
        foundExtensions.insert(match[1]);
      }
    }
  }

  for (const auto & ext : expectedExtensions)
  {
    ASSERT_TRUE(foundExtensions.count(ext)) << "Missing expected file extension: ." << ext;
  }
}

TEST(DumpTransformationTests, MultipleRunsGetSequentialNumbers)
{
  // Arrange — create a unique temp directory and pass it via the StatisticsCollector
  // so files don't pollute the project directory.
  auto tmpDirName = "jlm-dump-test-" + jlm::util::CreateRandomAlphanumericString(6);
  auto tmpDir = fs::temp_directory_path() / tmpDirName;
  fs::create_directories(tmpDir);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & rvsdg = rvsdgModule.Rvsdg();

  const auto functionType = FunctionType::Create({}, {});
  const auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  lambdaNode->finalize({});

  // Act — run the same transformation twice, expecting sequential numbers
  DumpTransformation::Config config;
  config.formats = { DumpTransformation::OutputFormat::Dot };

  jlm::util::StatisticsCollector statisticsCollector(jlm::util::StatisticsCollectorSettings(
      {},
      std::optional<jlm::util::FilePath>(tmpDir),
      "test-module"));

  {
    DumpTransformation dump(config);
    dump.setLabel("first");
    dump.setPassNumber(0);
    dump.Run(rvsdgModule, statisticsCollector);
  }
  {
    DumpTransformation dump(config);
    dump.setLabel("second");
    dump.setPassNumber(1);
    dump.Run(rvsdgModule, statisticsCollector);
  }

  // Assert — both files exist in the temp directory with sequential numbers.
  auto entries = fs::directory_iterator(tmpDir);
  int first_count = 0;
  int second_count = 0;
  for (const auto & entry : entries)
  {
    if (entry.is_regular_file())
    {
      std::string str = entry.path();

      std::regex first_pattern("-000-first\\.llvm\\.dot");
      std::regex second_pattern("-001-second\\.llvm\\.dot");

      if (std::regex_search(str, first_pattern))
        ++first_count;
      if (std::regex_search(str, second_pattern))
        ++second_count;
    }
  }

  EXPECT_EQ(first_count, 1) << "Missing expected file matching *-000-first.llvm.dot";
  EXPECT_EQ(second_count, 1) << "Missing expected file matching *-001-second.llvm.dot";
}

} // namespace jlm::hls

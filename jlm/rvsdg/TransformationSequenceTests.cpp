/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/Transformation.hpp>

class TestTransformation final : public jlm::rvsdg::Transformation
{
public:
  TestTransformation()
      : Transformation("TestTransformation")
  {}

  void
  Run(jlm::rvsdg::RvsdgModule & module,
      jlm::util::StatisticsCollector & statisticsCollector) override
  {}
};

class TestDotWriter final : public jlm::rvsdg::DotWriter
{
protected:
  void
  AnnotateTypeGraphNode(const jlm::rvsdg::Type & type, jlm::util::graph::Node & node) override
  {}

  void
  AnnotateGraphNode(
      const jlm::rvsdg::Node & rvsdgNode,
      jlm::util::graph::Node & node,
      jlm::util::graph::Graph * typeGraph) override
  {}

  void
  AnnotateEdge(const jlm::rvsdg::Input & rvsdgInput, jlm::util::graph::Edge & edge) override
  {}

  void
  AnnotateRegionArgument(
      const jlm::rvsdg::RegionArgument & regionArgument,
      jlm::util::graph::Node & node,
      jlm::util::graph::Graph * typeGraph) override
  {}
};

TEST(ArgumentTests, RvsdgDumping)
{
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  RvsdgModule rvsdgModule(FilePath("/tmp/mySource"));

  const StatisticsCollectorSettings statisticsCollectorSettings({}, FilePath("/tmp"), "moduleName");
  StatisticsCollector statisticsCollector(statisticsCollectorSettings);

  TestDotWriter dotWriter;
  auto testTransformation = std::make_shared<TestTransformation>();

  // Act
  TransformationSequence::CreateAndRun(
      rvsdgModule,
      statisticsCollector,
      { testTransformation },
      dotWriter,
      true);

  // Assert
  EXPECT_TRUE(std::filesystem::exists("/tmp/000-Pristine.dot"));
  EXPECT_TRUE(std::filesystem::exists("/tmp/001-AfterTestTransformation.dot"));
}

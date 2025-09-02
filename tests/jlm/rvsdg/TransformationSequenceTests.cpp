/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

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

static void
RvsdgDumping()
{
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  RvsdgModule rvsdgModule(FilePath("/tmp/mySource"));

  const StatisticsCollectorSettings statisticsCollectorSettings({}, FilePath("/tmp"), "moduleName");
  StatisticsCollector statisticsCollector(statisticsCollectorSettings);

  TestDotWriter dotWriter;
  TestTransformation testTransformation;

  // Act
  TransformationSequence::CreateAndRun(
      rvsdgModule,
      statisticsCollector,
      { &testTransformation },
      dotWriter,
      true);

  // Assert
  assert(std::filesystem::exists("/tmp/000-Pristine.dot"));
  assert(std::filesystem::exists("/tmp/001-AfterTestTransformation.dot"));
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/TransformationSequenceTests-RvsdgDumping", RvsdgDumping)

/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_DOTWRITER_HPP
#define JLM_LLVM_DOTWRITER_HPP

#include <jlm/rvsdg/region.hpp>
#include <jlm/util/GraphWriter.hpp>

namespace jlm::llvm::dot
{

class DotWriter
{
public:
  virtual ~DotWriter() noexcept;

  /**
   * Recursively converts a region and all sub-regions into graphs and sub-graphs.
   * All nodes in each region become InOutNodes, with edges showing data and state dependencies.
   * Arguments and results are represented using ArgumentNode and ResultNode, respectively.
   * All created nodes, inputs, and outputs, get associated to the rvsdg nodes, inputs and outputs.
   *
   * @param writer the GraphWriter to use
   * @param region the RVSDG region to recursively traverse
   * @param emitTypeGraph if true, an additional graph containing nodes for all types is emitted
   * @return a reference to the top-level graph corresponding to the region
   */
  util::graph::Graph &
  WriteGraphs(util::graph::Writer & writer, rvsdg::Region & region, bool emitTypeGraph);

protected:
  util::graph::Node &
  GetOrCreateTypeGraphNode(const rvsdg::Type & type, util::graph::Graph & typeGraph);

  virtual void
  AnnotateTypeGraphNode(const rvsdg::Type & type, util::graph::Node & node) = 0;

  virtual void
  AnnotateGraphNode(
      const rvsdg::Node & rvsdgNode,
      util::graph::Node & node,
      util::graph::Graph * typeGraph) = 0;

  virtual void
  AnnotateEdge(
    const rvsdg::Input & rvsdgInput,
    util::graph::Edge & edge) = 0;

  virtual void
  AnnotateRegionArgument(
      const rvsdg::RegionArgument & regionArgument,
      util::graph::Node & node,
      util::graph::Graph * typeGraph) = 0;

private:
  void
  CreateGraphNodes(
      util::graph::Graph & graph,
      rvsdg::Region & region,
      util::graph::Graph * typeGraph);

  void
  AttachNodeInput(util::graph::Port & inputPort, const rvsdg::Input & rvsdgInput);

  void
  AttachNodeOutput(
      util::graph::Port & outputPort,
      const rvsdg::Output & rvsdgOutput,
      util::graph::Graph * typeGraph);
};

class LlvmDotWriter final : public DotWriter
{
public:
  ~LlvmDotWriter() noexcept override;

protected:
  void
  AnnotateTypeGraphNode(const rvsdg::Type & type, util::graph::Node & node) override;

  void
  AnnotateGraphNode(
      const rvsdg::Node & rvsdgNode,
      util::graph::Node & node,
      util::graph::Graph * typeGraph) override;
  void
  AnnotateRegionArgument(
      const rvsdg::RegionArgument & regionArgument,
      util::graph::Node & node,
      util::graph::Graph * typeGraph) override;

  void
  AnnotateEdge(
    const rvsdg::Input & rvsdgInput,
    util::graph::Edge & edge) override;
};

}

#endif // JLM_LLVM_DOTWRITER_HPP

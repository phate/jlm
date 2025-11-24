/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_DOTWRITER_HPP
#define JLM_RVSDG_DOTWRITER_HPP

#include <jlm/rvsdg/region.hpp>
#include <jlm/util/GraphWriter.hpp>

namespace jlm::rvsdg
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
  WriteGraphs(util::graph::Writer & writer, Region & region, bool emitTypeGraph);

protected:
  util::graph::Node &
  GetOrCreateTypeGraphNode(const Type & type, util::graph::Graph & typeGraph);

  virtual void
  AnnotateTypeGraphNode(const Type & type, util::graph::Node & node);

  virtual void
  AnnotateGraphNode(
      const Node & rvsdgNode,
      util::graph::Node & node,
      util::graph::Graph * typeGraph);

  virtual void
  AnnotateEdge(const Input & rvsdgInput, util::graph::Edge & edge);

  virtual void
  AnnotateRegionArgument(
      const RegionArgument & regionArgument,
      util::graph::Node & node,
      util::graph::Graph * typeGraph);

private:
  void
  CreateGraphNodes(util::graph::Graph & graph, Region & region, util::graph::Graph * typeGraph);

  void
  AttachNodeInput(util::graph::Port & inputPort, const Input & rvsdgInput);

  void
  AttachNodeOutput(
      util::graph::Port & outputPort,
      const Output & rvsdgOutput,
      util::graph::Graph * typeGraph);
};

}

#endif

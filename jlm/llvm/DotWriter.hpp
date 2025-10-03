/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_DOTWRITER_HPP
#define JLM_LLVM_DOTWRITER_HPP

#include <jlm/rvsdg/DotWriter.hpp>

namespace jlm::llvm
{

class LlvmDotWriter : public rvsdg::DotWriter
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
  AnnotateEdge(const rvsdg::Input & rvsdgInput, util::graph::Edge & edge) override;

public:
  /**
   * Utility function for quickly dumping a region and its subregions as a Dot graph.
   * @param region the region to dump
   * @param os the output stream to write the graph to, e.g. std::cout or an std::ofstream
   */
  static void
  CreateAndDumpDot(rvsdg::Region & region, std::ostream & os);
};

}

#endif // JLM_LLVM_DOTWRITER_HPP

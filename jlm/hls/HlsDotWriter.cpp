/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/HlsDotWriter.hpp>
#include <jlm/hls/ir/hls.hpp>

namespace jlm::hls
{

HlsDotWriter::~HlsDotWriter() noexcept = default;

void
HlsDotWriter::AnnotateTypeGraphNode(const rvsdg::Type & type, util::graph::Node & node)
{
  auto & typeGraph = node.GetGraph();

  if (const auto bundleType = dynamic_cast<const BundleType *>(&type))
  {
    for (auto [_, elementType] : bundleType->elements_)
    {
      auto & elementTypeNode = GetOrCreateTypeGraphNode(*elementType, typeGraph);
      typeGraph.CreateDirectedEdge(elementTypeNode, node);
    }
  }
  else
  {
    LlvmDotWriter::AnnotateTypeGraphNode(type, node);
  }
}

}

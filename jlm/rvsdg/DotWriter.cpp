/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/DotWriter.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/strfmt.hpp>

namespace jlm::rvsdg
{

DotWriter::~DotWriter() noexcept = default;

/**
 * Creates a node in the \p typeGraph representing the given \p type,
 * or returns such a node if it has already been created.
 */
util::graph::Node &
DotWriter::GetOrCreateTypeGraphNode(const rvsdg::Type & type, util::graph::Graph & typeGraph)
{
  // If the type already has a corresponding node, return it
  if (auto * graphElement = typeGraph.GetElementFromProgramObject(type))
  {
    auto * node = reinterpret_cast<util::graph::Node *>(graphElement);
    JLM_ASSERT(node);
    return *node;
  }

  auto & node = typeGraph.CreateNode();
  node.SetProgramObject(type);
  node.SetLabel(type.debug_string());

  AnnotateTypeGraphNode(type, node);

  return node;
}

/**
 * Attaches the given GraphWriter port to the input in the RVSDG it represents.
 * Also adds an edge to the input port, from the node representing the input's origin.
 * @param inputPort the GraphWriter port representing the input
 * @param rvsdgInput the RVSDG input
 */
void
DotWriter::AttachNodeInput(util::graph::Port & inputPort, const rvsdg::Input & rvsdgInput)
{
  auto & graph = inputPort.GetGraph();
  inputPort.SetProgramObject(rvsdgInput);
  inputPort.SetLabel(util::strfmt(rvsdgInput.index()));

  // nodes are visited in topological order, so if the origin is an output, it will already exist
  if (auto originPort = reinterpret_cast<util::graph::Port *>(
          graph.GetElementFromProgramObject(*rvsdgInput.origin())))
  {
    auto & edge = graph.CreateDirectedEdge(*originPort, inputPort);
    AnnotateEdge(rvsdgInput, edge);
  }
}

/**
 * Attaches the given GraphWriter port to the output in RVSDG it represents.
 * Also adds information to the output about its type, using a reference to the type graph.
 * @param outputPort the GraphWriter port representing the output
 * @param rvsdgOutput the RVSDG output
 * @param typeGraph the type graph, or nullptr if the output's type should not be included
 */
void
DotWriter::AttachNodeOutput(
    util::graph::Port & outputPort,
    const rvsdg::Output & rvsdgOutput,
    util::graph::Graph * typeGraph)
{
  outputPort.SetProgramObject(rvsdgOutput);
  outputPort.SetLabel(util::strfmt(rvsdgOutput.index()));

  if (typeGraph)
    outputPort.SetAttributeGraphElement(
        "type",
        GetOrCreateTypeGraphNode(*rvsdgOutput.Type(), *typeGraph));
}

/**
 * Fill the given \p graph with nodes corresponding to the nodes of the given \p region.
 * If \p typeGraph is not nullptr, all rvsdg outputs get a type reference to the type graph.
 * If the type does not already exist in the type graph, it is created.
 */
void
DotWriter::CreateGraphNodes(
    util::graph::Graph & graph,
    rvsdg::Region & region,
    util::graph::Graph * typeGraph)
{
  graph.SetProgramObject(region);

  // Start by creating nodes for all the region arguments, and attaching them to the RVSDG outputs.
  for (size_t n = 0; n < region.narguments(); n++)
  {
    auto & node = graph.CreateArgumentNode();
    auto & argument = *region.argument(n);
    AttachNodeOutput(node, argument, typeGraph);

    // Include the index among the region's attributes
    node.SetAttribute("index", std::to_string(n));

    // Use the debug string as the label
    node.SetLabel(argument.debug_string());

    // If this argument corresponds to one of the structural node's inputs, reference it
    if (argument.input())
    {
      node.SetAttributeObject("input", *argument.input());
      // Include the local index of the node's input in the label
      node.AppendToLabel(util::strfmt("<- ", argument.input()->debug_string()), " ");
    }

    AnnotateRegionArgument(argument, node, typeGraph);
  }

  // Create a node for each node in the region in topological order.
  // Inputs expect the node representing their origin to exist before being visited.
  rvsdg::TopDownTraverser traverser(&region);
  for (const auto rvsdgNode : traverser)
  {
    auto & node = graph.CreateInOutNode(rvsdgNode->ninputs(), rvsdgNode->noutputs());
    node.SetLabel(rvsdgNode->DebugString());
    node.SetProgramObject(*rvsdgNode);

    for (size_t i = 0; i < rvsdgNode->ninputs(); i++)
      AttachNodeInput(node.GetInputPort(i), *rvsdgNode->input(i));

    for (size_t i = 0; i < rvsdgNode->noutputs(); i++)
      AttachNodeOutput(node.GetOutputPort(i), *rvsdgNode->output(i), typeGraph);

    // Structural nodes also have subgraphs
    if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(rvsdgNode))
    {
      for (size_t i = 0; i < structuralNode->nsubregions(); i++)
      {
        auto & subGraph = node.CreateSubgraph();
        CreateGraphNodes(subGraph, *structuralNode->subregion(i), typeGraph);
      }
    }

    AnnotateGraphNode(*rvsdgNode, node, typeGraph);
  }

  // Create result nodes for the region's results, and attach them to their origins
  for (size_t n = 0; n < region.nresults(); n++)
  {
    auto & node = graph.CreateResultNode();
    auto & result = *region.result(n);
    AttachNodeInput(node, result);

    // Include the index among the region's results
    node.SetAttribute("index", std::to_string(n));

    // Use the debug string as the label
    node.SetLabel(result.debug_string());

    // If this result corresponds to one of the structural node's outputs, reference it
    if (result.output())
    {
      node.SetAttributeObject("output", *result.output());
      // Include the local index of the node's output in the label
      node.AppendToLabel(util::strfmt("-> ", result.output()->debug_string()), " ");
    }
  }
}

util::graph::Graph &
DotWriter::WriteGraphs(util::graph::Writer & writer, rvsdg::Region & region, bool emitTypeGraph)
{
  util::graph::Graph * typeGraph = nullptr;
  if (emitTypeGraph)
  {
    typeGraph = &writer.CreateGraph();
    typeGraph->SetLabel("Type graph");
  }
  util::graph::Graph & rootGraph = writer.CreateGraph();
  rootGraph.SetLabel("RVSDG root graph");
  CreateGraphNodes(rootGraph, region, typeGraph);

  return rootGraph;
}

}

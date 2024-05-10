/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/backend/rvsdg2dot/rvsdg2dot.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/type.hpp>

namespace jlm::llvm::rvsdg2dot
{
/**
 * Creates a node in the \p typeGraph representing the given \p type,
 * or returns such a node if it has already been created.
 * The function is recursive, and will create nodes for subtypes of aggregate types.
 */
static util::Node &
GetOrCreateTypeGraphNode(const rvsdg::type & type, util::Graph & typeGraph)
{
  // If the type already has a corresponding node, return it
  if (auto * graphElement = typeGraph.GetElementFromProgramObject(&type))
  {
    auto * node = reinterpret_cast<util::Node *>(graphElement);
    JLM_ASSERT(node);
    return *node;
  }

  auto & node = typeGraph.CreateNode();
  node.SetProgramObject(&type);
  node.SetLabel(type.debug_string());

  // For all aggregate types, add edges from the nodes representing the children
  if (auto arrayType = dynamic_cast<const arraytype *>(&type))
  {
    auto & elementTypeNode = GetOrCreateTypeGraphNode(arrayType->element_type(), typeGraph);
    typeGraph.CreateDirectedEdge(elementTypeNode, node);
  }
  else if (auto structType = dynamic_cast<const StructType *>(&type))
  {
    auto & structDeclaration = structType->GetDeclaration();
    for (size_t n = 0; n < structDeclaration.NumElements(); n++)
    {
      auto & elementTypeNode = GetOrCreateTypeGraphNode(structDeclaration.GetElement(n), typeGraph);
      typeGraph.CreateDirectedEdge(elementTypeNode, node);
    }
  }
  else if (auto vectorType = dynamic_cast<const vectortype *>(&type))
  {
    auto & elementTypeNode = GetOrCreateTypeGraphNode(vectorType->type(), typeGraph);
    typeGraph.CreateDirectedEdge(elementTypeNode, node);
  }
  // TODO: Function types are also aggregate types (arguments and results)

  return node;
}

/**
 * Fill the given \p graph with nodes corresponding to the nodes of the given \p region.
 * If \p typeGraph is not nullptr, all rvsdg outputs get a type reference to the type graph.
 * If the type does not already exist in the type graph, it is created.
 */
static void
CreateGraphNodes(util::Graph & graph, rvsdg::region & region, util::Graph * typeGraph)
{
  graph.SetProgramObject(&region);

  // Connects an input port in the GraphWriter graph to an input in the RVSDG.
  // Also adds an edge from the input's origin to the input port in the graph.
  auto AttachInput = [&](util::Port & inputPort, const rvsdg::input & rvsdgInput)
  {
    inputPort.SetProgramObject(&rvsdgInput);

    if (auto originPort =
            reinterpret_cast<util::Port *>(graph.GetElementFromProgramObject(rvsdgInput.origin())))
    {
      graph.CreateDirectedEdge(*originPort, inputPort);
      if (rvsdg::is<MemoryStateType>(rvsdgInput.type()))
        graph.SetAttribute("color", util::Colors::Red);
      if (rvsdg::is<iostatetype>(rvsdgInput.type()))
        graph.SetAttribute("color", util::Colors::Green);
    }
  };

  // Connects an output port in the GraphWriter graph to an output in the RVSDG
  // Also adds a type attribute referencing a node in the typeGraph, if one exists.
  auto AttachOutput = [&](util::Port & outputPort, const rvsdg::output & rvsdgOutput)
  {
    outputPort.SetProgramObject(&rvsdgOutput);
    if (typeGraph)
      outputPort.SetAttributeGraphElement(
          "type",
          GetOrCreateTypeGraphNode(rvsdgOutput.type(), *typeGraph));
  };

  for (size_t n = 0; n < region.narguments(); n++)
    AttachOutput(graph.CreateArgumentNode(), *region.argument(n));

  rvsdg::topdown_traverser traverser(&region);
  for (const auto rvsdgNode : traverser)
  {
    auto & node = graph.CreateInOutNode(rvsdgNode->ninputs(), rvsdgNode->noutputs());
    node.SetLabel(rvsdgNode->operation().debug_string());
    node.SetProgramObject(rvsdgNode);

    for (size_t i = 0; i < rvsdgNode->ninputs(); i++)
      AttachInput(node.GetInputPort(i), *rvsdgNode->input(i));

    for (size_t i = 0; i < rvsdgNode->noutputs(); i++)
      AttachOutput(node.GetOutputPort(i), *rvsdgNode->output(i));

    // Structural nodes also have subgraphs
    if (auto structuralNode = dynamic_cast<const rvsdg::structural_node *>(rvsdgNode))
    {
      for (size_t i = 0; i < structuralNode->nsubregions(); i++)
      {
        auto & subGraph = node.CreateSubgraph();
        CreateGraphNodes(subGraph, *structuralNode->subregion(i), typeGraph);
        // FIXME: Attach the arguments and results from the subgraph to node's inputs and outputs
        // Requires special logic for each type of structural node
      }
    }
  }

  for (size_t n = 0; n < region.nresults(); n++)
    AttachInput(graph.CreateResultNode(), *region.result(n));
}

util::Graph &
WriteGraphs(util::GraphWriter & writer, rvsdg::region & region, bool emitTypeGraph)
{
  util::Graph * typeGraph = nullptr;
  if (emitTypeGraph)
  {
    typeGraph = &writer.CreateGraph();
    typeGraph->SetLabel("Type graph");
  }
  util::Graph & rootGraph = writer.CreateGraph();
  rootGraph.SetLabel("RVSDG root graph");
  CreateGraphNodes(rootGraph, region, typeGraph);

  return rootGraph;
}
}

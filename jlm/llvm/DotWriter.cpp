/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/DotWriter.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/type.hpp>
#include <jlm/rvsdg/UnitType.hpp>

namespace jlm::llvm::dot
{
/**
 * Creates a node in the \p typeGraph representing the given \p type,
 * or returns such a node if it has already been created.
 * The function is recursive, and will create nodes for subtypes of aggregate types.
 */
static util::graph::Node &
GetOrCreateTypeGraphNode(const rvsdg::Type & type, util::graph::Graph & typeGraph)
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

  // Some types get special handling, such as adding incoming edges from aggregate types
  if (rvsdg::is<rvsdg::StateType>(type) || rvsdg::is<rvsdg::BitType>(type)
      || rvsdg::is<PointerType>(type) || rvsdg::is<FloatingPointType>(type)
      || rvsdg::is<VariableArgumentType>(type) || rvsdg::is<rvsdg::UnitType>(type))
  {
    // No need to provide any information beyond the debug string
  }
  else if (auto arrayType = dynamic_cast<const ArrayType *>(&type))
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
  else if (const auto vectorType = dynamic_cast<const VectorType *>(&type))
  {
    auto & elementTypeNode = GetOrCreateTypeGraphNode(vectorType->type(), typeGraph);
    typeGraph.CreateDirectedEdge(elementTypeNode, node);
  }
  else if (auto functionType = dynamic_cast<const rvsdg::FunctionType *>(&type))
  {
    for (size_t n = 0; n < functionType->NumArguments(); n++)
    {
      auto & argumentTypeNode = GetOrCreateTypeGraphNode(functionType->ArgumentType(n), typeGraph);
      auto & edge = typeGraph.CreateDirectedEdge(argumentTypeNode, node);
      edge.SetAttribute("Argument#", util::strfmt(n));
    }
    for (size_t n = 0; n < functionType->NumResults(); n++)
    {
      auto & resultTypeNode = GetOrCreateTypeGraphNode(functionType->ResultType(n), typeGraph);
      auto & edge = typeGraph.CreateDirectedEdge(resultTypeNode, node);
      edge.SetAttribute("Result#", util::strfmt(n));
    }
  }
  else
  {
    JLM_UNREACHABLE("Unknown type");
  }

  return node;
}

/**
 * Attaches the given GraphWriter port to the input in the RVSDG it represents.
 * Also adds an edge to the input port, from the node representing the input's origin.
 * @param inputPort the GraphWriter port representing the input
 * @param rvsdgInput the RVSDG input
 */
static void
AttachNodeInput(util::graph::Port & inputPort, const rvsdg::Input & rvsdgInput)
{
  auto & graph = inputPort.GetGraph();
  inputPort.SetProgramObject(rvsdgInput);

  // nodes are visited in topological order, so if the origin is an output, it will already exist
  if (auto originPort = reinterpret_cast<util::graph::Port *>(
          graph.GetElementFromProgramObject(*rvsdgInput.origin())))
  {
    auto & edge = graph.CreateDirectedEdge(*originPort, inputPort);
    if (rvsdg::is<MemoryStateType>(rvsdgInput.Type()))
      edge.SetAttribute("color", util::graph::Colors::Red);
    if (rvsdg::is<IOStateType>(rvsdgInput.Type()))
      edge.SetAttribute("color", util::graph::Colors::Green);
  }
}

/**
 * Attaches the given GraphWriter port to the output in RVSDG it represents.
 * Also adds information to the output about its type, using a reference to the type graph.
 * @param outputPort the GraphWriter port representing the output
 * @param rvsdgOutput the RVSDG output
 * @param typeGraph the type graph, or nullptr if the output's type should not be included
 */
static void
AttachNodeOutput(
    util::graph::Port & outputPort,
    const rvsdg::Output & rvsdgOutput,
    util::graph::Graph * typeGraph)
{
  outputPort.SetProgramObject(rvsdgOutput);
  if (typeGraph)
    outputPort.SetAttributeGraphElement(
        "type",
        GetOrCreateTypeGraphNode(*rvsdgOutput.Type(), *typeGraph));
}

/**
 * Some RVSDG arguments can have extra attributes.
 * This function handles adding them to the output graph.
 *
 * @param rvsdgArgument the RVSDG argument being represented
 * @param node the output graph node representing it
 * @param typeGraph the optional type graph, used for dumping types
 */
static void
SetAdditionalArgumentAttributes(
    const rvsdg::RegionArgument & rvsdgArgument,
    util::graph::Node & node,
    util::graph::Graph * typeGraph)
{
  // If the argument is a GraphImport, include extra type and linkage data
  if (const auto graphImport = dynamic_cast<const GraphImport *>(&rvsdgArgument))
  {
    node.SetAttribute("linkage", ToString(graphImport->Linkage()));
    if (typeGraph)
    {
      // The output of a GraphImport is always a PointerType
      // Expose the underlying type as a separate attribute
      auto & valueTypeNode = GetOrCreateTypeGraphNode(*graphImport->ValueType(), *typeGraph);
      node.SetAttributeGraphElement("valueType", valueTypeNode);
    }
  }
}

/**
 * Some RVSDG nodes can have extra attributes.
 * This function handles adding them to the output graph.
 *
 * @param rvsdgNode the RVSDG node being represented
 * @param node the output graph node representing it
 * @param typeGraph the optional type graph, used for dumping types
 */
static void
SetAdditionalNodeAttributes(
    const rvsdg::Node & rvsdgNode,
    util::graph::Node & node,
    util::graph::Graph * typeGraph)
{
  node.SetAttribute("NodeId", util::strfmt(rvsdgNode.GetNodeId()));

  if (const auto delta = dynamic_cast<const rvsdg::DeltaNode *>(&rvsdgNode))
  {
    if (auto op = dynamic_cast<const llvm::DeltaOperation *>(&delta->GetOperation()))
    {
      node.SetAttribute("linkage", ToString(op->linkage()));
      node.SetAttribute("constant", op->constant() ? "true" : "false");
    }

    if (typeGraph)
    {
      // The output of a delta node is always a PointerType
      // Expose the underlying type as a separate attribute
      auto & typeNode = GetOrCreateTypeGraphNode(*delta->GetOperation().Type(), *typeGraph);
      node.SetAttributeGraphElement("type", typeNode);
    }
  }
}

/**
 * Fill the given \p graph with nodes corresponding to the nodes of the given \p region.
 * If \p typeGraph is not nullptr, all rvsdg outputs get a type reference to the type graph.
 * If the type does not already exist in the type graph, it is created.
 */
static void
CreateGraphNodes(util::graph::Graph & graph, rvsdg::Region & region, util::graph::Graph * typeGraph)
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

    SetAdditionalArgumentAttributes(argument, node, typeGraph);
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

    SetAdditionalNodeAttributes(*rvsdgNode, node, typeGraph);
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
WriteGraphs(util::graph::Writer & writer, rvsdg::Region & region, bool emitTypeGraph)
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

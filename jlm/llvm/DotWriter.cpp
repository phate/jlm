/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/DotWriter.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/UnitType.hpp>

namespace jlm::llvm
{

LlvmDotWriter::~LlvmDotWriter() noexcept = default;

void
LlvmDotWriter::AnnotateTypeGraphNode(const rvsdg::Type & type, util::graph::Node & node)
{
  auto & typeGraph = node.GetGraph();

  // Some types get special handling, such as adding incoming edges from aggregate types
  if (type.Kind() == rvsdg::TypeKind::State || rvsdg::is<rvsdg::BitType>(type)
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
}

void
LlvmDotWriter::AnnotateEdge(const rvsdg::Input & rvsdgInput, util::graph::Edge & edge)
{
  if (rvsdg::is<MemoryStateType>(rvsdgInput.Type()))
    edge.SetAttribute("color", util::graph::Colors::Red);
  if (rvsdg::is<IOStateType>(rvsdgInput.Type()))
    edge.SetAttribute("color", util::graph::Colors::Green);
}

/**
 * Some RVSDG arguments can have extra attributes.
 * This function handles adding them to the output graph.
 *
 * @param rvsdgArgument the RVSDG argument being represented
 * @param node the output graph node representing it
 * @param typeGraph the optional type graph, used for dumping types
 */
void
LlvmDotWriter::AnnotateRegionArgument(
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
void
LlvmDotWriter::AnnotateGraphNode(
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

}

/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/GraphWriter.hpp>
#include <jlm/util/strfmt.hpp>

#include <cassert>
#include <sstream>

static bool
StringContains(const std::string & haystack, const std::string & needle)
{
  return haystack.find(needle) != std::string::npos;
}

static void
TestGraphElement()
{
  using namespace jlm::util;
  GraphWriter writer;
  auto & graph = writer.CreateGraph();

  // Test labels
  graph.SetLabel("Test");
  assert(graph.GetLabel() == "Test");
  assert(graph.HasLabel());
  graph.SetLabel("");
  assert(!graph.HasLabel());
  assert(graph.GetLabelOr("default") == std::string("default"));

  // Test assigning a program object to a graph element
  int myInt = 0;
  graph.SetProgramObject(&myInt);
  assert(graph.GetProgramObject() == reinterpret_cast<uintptr_t>(&myInt));

  // Set attributes
  graph.SetAttribute("color", "\"brown\"");
  graph.SetAttributeGraphElement("graph", graph);
  graph.SetAttributeObject("another graph", &myInt);

  // Finalizing and getting a unique id
  assert(!graph.IsFinalized());
  graph.Finalize();
  assert(graph.IsFinalized());
  assert(graph.GetUniqueIdSuffix() == 0);
  assert(graph.GetFullId() == "graph0");

  // Attribute printing
  std::ostringstream out;
  graph.OutputAttributes(out, AttributeOutputFormat::SpaceSeparatedList);
  auto attributes = out.str();
  assert(StringContains(attributes, "color=\"\\\"brown\\\"\""));
  assert(StringContains(attributes, "graph=graph0"));
  assert(StringContains(attributes, "\"another graph\"=graph0"));

  // Also test HTML attribute escaping
  out = std::ostringstream();
  graph.OutputAttributes(out, AttributeOutputFormat::HTMLAttributes);
  attributes = out.str();
  assert(StringContains(attributes, "color=\"&quot;brown&quot;\""));
  assert(StringContains(attributes, "another-graph=graph0"));
}

static void
TestNode()
{
  using namespace jlm::util;
  GraphWriter writer;
  auto & graph = writer.CreateGraph();

  auto & node = graph.CreateNode();
  assert(&node.GetNode() == &node);
  assert(&node.GetGraph() == &graph);

  node.SetLabel("MyNode");

  node.Finalize();

  std::ostringstream out;
  node.Output(out, GraphOutputFormat::ASCII, 0);
  auto string = out.str();
  assert(StringContains(string, "MyNode"));
}

static void
TestASCIIEdges()
{
  using namespace jlm::util;
  GraphWriter writer;
  auto & graph = writer.CreateGraph();

  auto & node0 = graph.CreateNode();
  auto & node1 = graph.CreateNode();
  auto & node2 = graph.CreateNode();

  node0.SetLabel("NODE0");
  node1.SetLabel("NODE1");
  node2.SetLabel("NODE2");

  graph.CreateDirectedEdge(node0, node1);
  graph.CreateDirectedEdge(node0, node2);
  graph.CreateDirectedEdge(node1, node2);

  graph.Finalize();

  std::ostringstream out;
  node0.Output(out, GraphOutputFormat::ASCII, 0);
  node1.Output(out, GraphOutputFormat::ASCII, 0);
  node2.Output(out, GraphOutputFormat::ASCII, 0);

  auto string = out.str();
  assert(StringContains(string, "node0:NODE0"));
  assert(StringContains(string, "node1:NODE1<-node0"));
  assert(StringContains(string, "NODE2<-[node0, node1]"));
}

static void
TestInOutNode()
{
  using namespace jlm::util;
  GraphWriter writer;
  auto & graph = writer.CreateGraph();

  auto & node = graph.CreateInOutNode(2, 3);
  assert(node.NumInputPorts() == 2);
  assert(node.NumOutputPorts() == 3);

  node.SetLabel("InOutNode");

  graph.CreateDirectedEdge(node.GetOutputPort(2), node.GetInputPort(0));

  // Also test subgraphs, and connecting argument nodes and result nodes to outside ports
  auto & subgraph = node.CreateSubgraph();
  assert(node.NumSubgraphs() == 1);
  auto & argumentNode = subgraph.CreateArgumentNode();
  argumentNode.SetLabel("CTX");
  argumentNode.SetOutsideSource(node.GetInputPort(0));
  auto & resultNode = subgraph.CreateResultNode();
  resultNode.SetLabel("RETURN");
  resultNode.SetOutsideDestination(node.GetOutputPort(0));

  subgraph.CreateDirectedEdge(argumentNode, resultNode);

  graph.Finalize();

  std::ostringstream out;
  node.Output(out, GraphOutputFormat::ASCII, 0);
  auto string = out.str();
  assert(StringContains(string, "o0, o1, o2 := InOutNode o2, []"));

  // Check that the subgraph is also printed
  assert(StringContains(string, "ARG a0:CTX <= o2"));
  assert(StringContains(string, "RES a0:RETURN => o0"));
}

static void
TestEdge()
{
  using namespace jlm::util;
  GraphWriter writer;
  auto & graph = writer.CreateGraph();

  auto & node0 = graph.CreateNode();
  auto & node1 = graph.CreateNode();
  auto & node2 = graph.CreateNode();

  auto & edge0 = graph.CreateDirectedEdge(node0, node1);
  auto & edge1 = graph.CreateUndirectedEdge(node1, node2);

  assert(&edge0.GetFrom() == &node0);
  assert(&edge0.GetTo() == &node1);
  assert(edge0.IsDirected());
  assert(&edge1.GetFrom() == &node1);
  assert(&edge1.GetTo() == &node2);
  assert(!edge1.IsDirected());

  assert(&edge0.GetOtherEnd(node0) == &node1);
  assert(&edge0.GetOtherEnd(node1) == &node0);

  edge0.SetAttribute("color", Colors::Red);

  graph.Finalize();

  std::ostringstream out0;
  edge0.OutputDot(out0, 0);
  auto string0 = out0.str();

  assert(StringContains(string0, "node0 -> node1"));
  assert(StringContains(string0, jlm::util::strfmt("color=\"", Colors::Red, "\"")));

  std::ostringstream out1;
  edge1.OutputDot(out1, 0);
  auto string1 = out1.str();
  assert(StringContains(string1, "node1 -> node2"));
  assert(StringContains(string1, jlm::util::strfmt("dir=none")));
}

static void
TestGraph()
{
  using namespace jlm::util;
  GraphWriter writer;
  auto & graph = writer.CreateGraph();
  graph.SetLabel("My Graph");

  assert(&graph.GetGraphWriter() == &writer);
  auto & node = graph.CreateNode();
  auto & argumentNode = graph.CreateArgumentNode();
  auto & resultNode = graph.CreateResultNode();
  auto & inOutNode = graph.CreateInOutNode(1, 1);

  auto & subgraph = inOutNode.CreateSubgraph();
  assert(subgraph.IsSubgraph());
  assert(!graph.IsSubgraph());

  int myInt;
  node.SetProgramObject(&myInt);
  assert(&graph.GetFromProgramObject<Node>(&myInt) == &node);

  graph.SetAttributeObject("friend", &myInt);
  graph.SetAttributeGraphElement("foe", argumentNode);

  graph.Finalize();
  assert(node.IsFinalized());
  assert(argumentNode.IsFinalized());
  assert(resultNode.IsFinalized());
  assert(inOutNode.IsFinalized());
  assert(subgraph.IsFinalized());

  std::ostringstream out;
  graph.Output(out, jlm::util::GraphOutputFormat::Dot, 0);
  auto string = out.str();

  assert(StringContains(string, "label=\"My Graph\""));

  // Nodes referred to in attributes
  assert(StringContains(string, "friend=node0"));
  assert(StringContains(string, "foe=a0"));

  // Make sure the other nodes are also mentioned
  assert(StringContains(string, "r0"));
  assert(StringContains(string, "node1"));
}

static void
TestGraphWriterClass()
{
  using namespace jlm::util;
  GraphWriter writer;
  auto & graph0 = writer.CreateGraph();
  auto & graph1 = writer.CreateGraph();

  auto & node0 = graph0.CreateNode();
  auto & node1 = graph1.CreateNode();

  int myInt;
  node1.SetProgramObject(&myInt);

  assert(writer.GetElementFromProgramObject(reinterpret_cast<uintptr_t>(&myInt)) == &node1);

  // Refer to program objects mapped to elements in other graphs
  node0.SetAttributeObject("friend", &myInt);

  std::ostringstream out;
  writer.OutputAllGraphs(out, GraphOutputFormat::Dot);
  auto string = out.str();

  assert(graph0.IsFinalized());
  assert(graph1.IsFinalized());

  assert(node0.GetFullId() == "node0");
  assert(node1.GetFullId() == "node1");

  assert(StringContains(string, "friend=node1"));
}

static int
TestGraphWriter()
{
  TestGraphElement();
  TestNode();
  TestASCIIEdges();
  TestInOutNode();
  TestEdge();
  TestGraph();
  TestGraphWriterClass();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestGraphWriter", TestGraphWriter)

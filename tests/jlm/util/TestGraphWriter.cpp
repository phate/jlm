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

  // Test label appending
  graph.AppendToLabel("Text");
  assert(graph.GetLabel() == "Text");
  graph.AppendToLabel("Text2", "\n");
  assert(graph.GetLabel() == "Text\nText2");

  // Test assigning a program object to a graph element
  int myInt = 0;
  graph.SetProgramObject(myInt);
  assert(graph.GetProgramObject() == reinterpret_cast<uintptr_t>(&myInt));

  // Set attributes
  graph.SetAttribute("color", "\"dark\nbrown\"");
  graph.SetAttribute("taste", "sweet");
  graph.SetAttributeGraphElement("graph", graph);
  graph.SetAttributeObject("another graph", myInt);

  // Check getting attributes
  assert(graph.HasAttribute("taste"));
  assert(graph.GetAttributeString("taste") == "sweet");
  assert(!graph.GetAttributeString("not-an-attribute"));
  assert(graph.GetAttributeGraphElement("graph") == &graph);
  assert(graph.GetAttributeObject("another graph") == reinterpret_cast<uintptr_t>(&myInt));
  // Also check that one can get GraphElements based on the program object they represent
  assert(graph.GetAttributeGraphElement("another graph") == &graph);

  // Test removing attributes
  assert(graph.RemoveAttribute("taste"));
  assert(!graph.HasAttribute("taste"));
  // Removing the attribute again returns false
  assert(!graph.RemoveAttribute("taste"));

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
  assert(StringContains(attributes, "color=\"\\\"dark\\nbrown\\\"\""));
  assert(StringContains(attributes, "graph=graph0"));
  assert(StringContains(attributes, "\"another graph\"=graph0"));

  // Also test HTML attribute escaping
  out = std::ostringstream();
  graph.OutputAttributes(out, AttributeOutputFormat::HTMLAttributes);
  attributes = out.str();
  assert(StringContains(attributes, "color=\"&quot;dark\nbrown&quot;\""));
  assert(StringContains(attributes, "another-graph=\"graph0\""));
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

  node.SetShape(Node::Shape::Rectangle);
  assert(node.HasAttribute("shape"));

  node.Finalize();

  std::ostringstream out;
  node.Output(out, GraphOutputFormat::ASCII, 0);
  auto string = out.str();
  assert(StringContains(string, "MyNode"));

  std::ostringstream out2;
  node.Output(out2, GraphOutputFormat::Dot, 0);
  auto string2 = out2.str();
  assert(StringContains(string2, "label=MyNode"));
  assert(StringContains(string2, "shape=rect"));
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

  node.SetLabel("My\nInOutNode");

  graph.CreateDirectedEdge(node.GetOutputPort(2), node.GetInputPort(0));

  // Also test subgraphs, and connecting argument nodes and result nodes to outside ports
  auto & subgraph = node.CreateSubgraph();
  assert(node.NumSubgraphs() == 1);
  assert(&node.GetSubgraph(0) == &subgraph);
  auto & argumentNode = subgraph.CreateArgumentNode();
  argumentNode.SetLabel("CTX");
  argumentNode.SetOutsideSource(node.GetInputPort(0));
  auto & resultNode = subgraph.CreateResultNode();
  resultNode.SetLabel("RETURN");
  resultNode.SetOutsideDestination(node.GetOutputPort(0));

  subgraph.CreateDirectedEdge(argumentNode, resultNode);

  graph.Finalize();
  assert(node.IsFinalized());
  assert(subgraph.IsFinalized());

  std::ostringstream out;
  node.Output(out, GraphOutputFormat::ASCII, 0);
  auto string = out.str();
  assert(StringContains(string, "out0, out1, out2 := \"My\\nInOutNode\" out2, []"));

  // Check that the subgraph is also printed
  assert(StringContains(string, "ARG arg0:CTX <= out2"));
  assert(StringContains(string, "RES arg0:RETURN => out0"));

  // Check that HTML labels with newlines turn into <BR/>
  std::ostringstream out2;
  node.Output(out2, GraphOutputFormat::Dot, 0);
  auto string0 = out2.str();
  assert(StringContains(string0, "My<BR/>InOutNode"));
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

  assert(graph.NumEdges() == 2);
  assert(&graph.GetEdge(0) == &edge0);

  assert(graph.GetEdgeBetween(node0, node1) == &edge0);
  assert(graph.GetEdgeBetween(node1, node2) == &edge1);
  assert(graph.GetEdgeBetween(node2, node0) == nullptr);

  edge0.SetAttribute("color", Colors::Red);

  auto & edge2 = graph.CreateUndirectedEdge(node2, node0);
  edge2.SetArrowHead("odot");
  edge2.SetArrowTail("normal");
  edge2.SetStyle(Edge::Style::Tapered);

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

  std::ostringstream out2;
  edge2.OutputDot(out2, 0);
  auto string2 = out2.str();
  assert(StringContains(string2, "dir=both"));
  assert(StringContains(string2, "arrowhead=odot"));
  assert(StringContains(string2, "arrowtail=normal"));
  assert(StringContains(string2, "style=tapered"));
}

static void
TestGraphCreateNodes()
{
  using namespace jlm::util;
  GraphWriter writer;
  auto & graph = writer.CreateGraph();

  // Test node creation and count
  assert(graph.NumNodes() == 0);
  auto & node = graph.CreateNode();
  assert(graph.NumNodes() == 1);
  assert(&graph.GetNode(0) == &node);

  // Test InOutNode creation and count
  auto & inOutNode = graph.CreateInOutNode(1, 1);
  assert(graph.NumNodes() == 2);
  assert(&graph.GetNode(1) == &inOutNode);

  // Test argument node creation and count
  assert(graph.NumArgumentNodes() == 0);
  auto & argumentNode = graph.CreateArgumentNode();
  assert(graph.NumArgumentNodes() == 1);
  assert(&graph.GetArgumentNode(0) == &argumentNode);

  // Test result node creation and count
  assert(graph.NumResultNodes() == 0);
  auto & resultNode = graph.CreateResultNode();
  assert(graph.NumResultNodes() == 1);
  assert(&graph.GetResultNode(0) == &resultNode);

  // Test finalizing reaching every node
  graph.Finalize();
  assert(node.IsFinalized());
  assert(argumentNode.IsFinalized());
  assert(resultNode.IsFinalized());
  assert(inOutNode.IsFinalized());
}

static void
TestGraphAttributes()
{
  using namespace jlm::util;
  GraphWriter writer;
  auto & graph = writer.CreateGraph();
  graph.SetLabel("My Graph");

  assert(&graph.GetGraphWriter() == &writer);
  auto & node = graph.CreateNode();

  // Test associating a GraphElement with a pointer, and retrieving it
  int myInt = 6;
  node.SetProgramObject(myInt);
  assert(&graph.GetFromProgramObject<Node>(myInt) == &node);

  // Set some attributes, to test that they appear in the final output
  graph.SetAttributeObject("friend", myInt);
  graph.SetAttributeGraphElement("foe", graph);

  graph.Finalize();

  // Test that the Dot output of the graph contains everything specified
  std::ostringstream out;
  graph.Output(out, jlm::util::GraphOutputFormat::Dot, 0);
  auto string = out.str();

  assert(StringContains(string, "label=\"My Graph\""));

  // Nodes referred to in attributes
  assert(StringContains(string, "friend=node0"));
  assert(StringContains(string, "foe=graph0"));
}

static void
TestGraphWriterClass()
{
  using namespace jlm::util;
  GraphWriter writer;

  auto & graph0 = writer.CreateGraph();
  auto & graph1 = writer.CreateGraph();
  assert(writer.NumGraphs() == 2);
  assert(&writer.GetGraph(0) == &graph0);

  auto & node0 = graph0.CreateNode();
  auto & node1 = graph1.CreateNode();

  // Test retrieving a GraphElement from its associated program object pointer
  int myInt = 12;
  node1.SetProgramObject(myInt);
  assert(writer.GetElementFromProgramObject(reinterpret_cast<uintptr_t>(&myInt)) == &node1);

  // Refer to program objects mapped to elements in other graphs
  node0.SetAttributeObject("friend", myInt);

  // Render all the graphs to dot, which first finalizes the graphs to assign unique IDs
  std::ostringstream out;
  writer.OutputAllGraphs(out, GraphOutputFormat::Dot);
  auto string = out.str();

  assert(graph0.IsFinalized());
  assert(graph1.IsFinalized());

  assert(node0.GetFullId() == "node0");
  assert(node1.GetFullId() == "node1");

  assert(StringContains(string, "friend=node1"));
}

static void
TestGraphWriter()
{
  TestGraphElement();
  TestNode();
  TestASCIIEdges();
  TestInOutNode();
  TestEdge();
  TestGraphCreateNodes();
  TestGraphAttributes();
  TestGraphWriterClass();
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestGraphWriter", TestGraphWriter)

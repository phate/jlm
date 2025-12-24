/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/util/GraphWriter.hpp>
#include <jlm/util/strfmt.hpp>

#include <sstream>

static bool
StringContains(const std::string & haystack, const std::string & needle)
{
  return haystack.find(needle) != std::string::npos;
}

TEST(GraphWriterTests, TestGraphElement)
{
  using namespace jlm::util;
  using namespace jlm::util::graph;
  Writer writer;
  auto & graph = writer.CreateGraph();

  // Test labels
  graph.SetLabel("Test");
  EXPECT_EQ(graph.GetLabel(), "Test");
  EXPECT_TRUE(graph.HasLabel());
  graph.SetLabel("");
  EXPECT_FALSE(graph.HasLabel());
  EXPECT_EQ(graph.GetLabelOr("default"), std::string("default"));

  // Test label appending
  graph.AppendToLabel("Text");
  EXPECT_EQ(graph.GetLabel(), "Text");
  graph.AppendToLabel("Text2", "\n");
  EXPECT_EQ(graph.GetLabel(), "Text\nText2");

  // Test assigning a program object to a graph element
  int myInt = 0;
  graph.SetProgramObject(myInt);
  EXPECT_EQ(graph.GetProgramObject(), reinterpret_cast<uintptr_t>(&myInt));

  // Set attributes
  graph.SetAttribute("color", "\"dark\nbrown\"");
  graph.SetAttribute("taste", "sweet");
  graph.SetAttributeGraphElement("graph", graph);
  graph.SetAttributeObject("another graph", myInt);

  // Check getting attributes
  EXPECT_TRUE(graph.HasAttribute("taste"));
  EXPECT_EQ(graph.GetAttributeString("taste"), "sweet");
  EXPECT_FALSE(graph.GetAttributeString("not-an-attribute"));
  EXPECT_EQ(graph.GetAttributeGraphElement("graph"), &graph);
  EXPECT_EQ(graph.GetAttributeObject("another graph"), reinterpret_cast<uintptr_t>(&myInt));
  // Also check that one can get GraphElements based on the program object they represent
  EXPECT_EQ(graph.GetAttributeGraphElement("another graph"), &graph);

  // Test removing attributes
  EXPECT_TRUE(graph.RemoveAttribute("taste"));
  EXPECT_FALSE(graph.HasAttribute("taste"));
  // Removing the attribute again returns false
  EXPECT_FALSE(graph.RemoveAttribute("taste"));

  // Finalizing and getting a unique id
  EXPECT_FALSE(graph.IsFinalized());
  graph.Finalize();
  EXPECT_TRUE(graph.IsFinalized());
  EXPECT_EQ(graph.GetUniqueIdSuffix(), 0);
  EXPECT_EQ(graph.GetFullId(), "graph0");

  // Attribute printing
  std::ostringstream out;
  graph.OutputAttributes(out, AttributeOutputFormat::SpaceSeparatedList);
  auto attributes = out.str();
  EXPECT_TRUE(StringContains(attributes, "color=\"\\\"dark\\nbrown\\\"\""));
  EXPECT_TRUE(StringContains(attributes, "graph=graph0"));
  EXPECT_TRUE(StringContains(attributes, "\"another graph\"=graph0"));

  // Also test HTML attribute escaping
  out = std::ostringstream();
  graph.OutputAttributes(out, AttributeOutputFormat::HTMLAttributes);
  attributes = out.str();
  EXPECT_TRUE(StringContains(attributes, "color=\"&quot;dark\nbrown&quot;\""));
  EXPECT_TRUE(StringContains(attributes, "another-graph=\"graph0\""));
}

TEST(GraphWriterTests, TestNode)
{
  using namespace jlm::util;
  using namespace jlm::util::graph;
  Writer writer;
  auto & graph = writer.CreateGraph();

  auto & node = graph.CreateNode();
  EXPECT_EQ(&node.GetNode(), &node);
  EXPECT_EQ(&node.GetGraph(), &graph);

  node.SetLabel("MyNode");

  node.SetShape(Node::Shape::Rectangle);
  EXPECT_TRUE(node.HasAttribute("shape"));

  node.Finalize();

  std::ostringstream out;
  node.Output(out, OutputFormat::ASCII, 0);
  auto string = out.str();
  EXPECT_TRUE(StringContains(string, "MyNode"));

  std::ostringstream out2;
  node.Output(out2, OutputFormat::Dot, 0);
  auto string2 = out2.str();
  EXPECT_TRUE(StringContains(string2, "label=MyNode"));
  EXPECT_TRUE(StringContains(string2, "shape=rect"));
}

TEST(GraphWriterTests, TestASCIIEdges)
{
  using namespace jlm::util;
  using namespace jlm::util::graph;
  Writer writer;
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
  node0.Output(out, OutputFormat::ASCII, 0);
  node1.Output(out, OutputFormat::ASCII, 0);
  node2.Output(out, OutputFormat::ASCII, 0);

  auto string = out.str();
  EXPECT_TRUE(StringContains(string, "node0:NODE0"));
  EXPECT_TRUE(StringContains(string, "node1:NODE1<-node0"));
  EXPECT_TRUE(StringContains(string, "NODE2<-[node0, node1]"));
}

TEST(GraphWriterTests, TestInOutNode)
{
  using namespace jlm::util;
  using namespace jlm::util::graph;
  Writer writer;
  auto & graph = writer.CreateGraph();

  auto & node = graph.CreateInOutNode(2, 3);
  EXPECT_EQ(node.NumInputPorts(), 2);
  EXPECT_EQ(node.NumOutputPorts(), 3);

  node.SetLabel("My\nInOutNode");

  graph.CreateDirectedEdge(node.GetOutputPort(2), node.GetInputPort(0));

  // Also test subgraphs, and connecting argument nodes and result nodes to outside ports
  auto & subgraph = node.CreateSubgraph();
  EXPECT_EQ(node.NumSubgraphs(), 1);
  EXPECT_EQ(&node.GetSubgraph(0), &subgraph);
  auto & argumentNode = subgraph.CreateArgumentNode();
  argumentNode.SetLabel("CTX");
  argumentNode.SetOutsideSource(node.GetInputPort(0));
  auto & resultNode = subgraph.CreateResultNode();
  resultNode.SetLabel("RETURN");
  resultNode.SetOutsideDestination(node.GetOutputPort(0));

  subgraph.CreateDirectedEdge(argumentNode, resultNode);

  graph.Finalize();
  EXPECT_TRUE(node.IsFinalized());
  EXPECT_TRUE(subgraph.IsFinalized());

  std::ostringstream out;
  node.Output(out, OutputFormat::ASCII, 0);
  auto string = out.str();
  EXPECT_TRUE(StringContains(string, "out0, out1, out2 := \"My\\nInOutNode\" out2, []"));

  // Check that the subgraph is also printed
  EXPECT_TRUE(StringContains(string, "ARG arg0:CTX <= out2"));
  EXPECT_TRUE(StringContains(string, "RES arg0:RETURN => out0"));

  // Check that HTML labels with newlines turn into <BR/>
  std::ostringstream out2;
  node.Output(out2, OutputFormat::Dot, 0);
  auto string0 = out2.str();
  EXPECT_TRUE(StringContains(string0, "My<BR/>InOutNode"));
}

TEST(GraphWriterTests, TestEdge)
{
  using namespace jlm::util;
  using namespace jlm::util::graph;
  Writer writer;
  auto & graph = writer.CreateGraph();

  auto & node0 = graph.CreateNode();
  auto & node1 = graph.CreateNode();
  auto & node2 = graph.CreateNode();

  auto & edge0 = graph.CreateDirectedEdge(node0, node1);
  auto & edge1 = graph.CreateUndirectedEdge(node1, node2);

  EXPECT_EQ(&edge0.GetFrom(), &node0);
  EXPECT_EQ(&edge0.GetTo(), &node1);
  EXPECT_TRUE(edge0.IsDirected());
  EXPECT_EQ(&edge1.GetFrom(), &node1);
  EXPECT_EQ(&edge1.GetTo(), &node2);
  EXPECT_TRUE(!edge1.IsDirected());

  EXPECT_EQ(&edge0.GetOtherEnd(node0), &node1);
  EXPECT_EQ(&edge0.GetOtherEnd(node1), &node0);

  EXPECT_EQ(graph.NumEdges(), 2);
  EXPECT_EQ(&graph.GetEdge(0), &edge0);

  EXPECT_EQ(graph.GetEdgeBetween(node0, node1), &edge0);
  EXPECT_EQ(graph.GetEdgeBetween(node1, node2), &edge1);
  EXPECT_EQ(graph.GetEdgeBetween(node2, node0), nullptr);

  edge0.SetAttribute("color", Colors::Red);

  auto & edge2 = graph.CreateUndirectedEdge(node2, node0);
  edge2.SetArrowHead("odot");
  edge2.SetArrowTail("normal");
  edge2.SetStyle(Edge::Style::Tapered);

  graph.Finalize();

  std::ostringstream out0;
  edge0.OutputDot(out0, 0);
  auto string0 = out0.str();

  EXPECT_TRUE(StringContains(string0, "node0 -> node1"));
  EXPECT_TRUE(StringContains(string0, jlm::util::strfmt("color=\"", Colors::Red, "\"")));

  std::ostringstream out1;
  edge1.OutputDot(out1, 0);
  auto string1 = out1.str();
  EXPECT_TRUE(StringContains(string1, "node1 -> node2"));
  EXPECT_TRUE(StringContains(string1, jlm::util::strfmt("dir=none")));

  std::ostringstream out2;
  edge2.OutputDot(out2, 0);
  auto string2 = out2.str();
  EXPECT_TRUE(StringContains(string2, "dir=both"));
  EXPECT_TRUE(StringContains(string2, "arrowhead=odot"));
  EXPECT_TRUE(StringContains(string2, "arrowtail=normal"));
  EXPECT_TRUE(StringContains(string2, "style=tapered"));
}

TEST(GraphWriterTests, TestGraphCreateNodes)
{
  using namespace jlm::util;
  using namespace jlm::util::graph;
  Writer writer;
  auto & graph = writer.CreateGraph();

  // Test node creation and count
  EXPECT_EQ(graph.NumNodes(), 0);
  auto & node = graph.CreateNode();
  EXPECT_EQ(graph.NumNodes(), 1);
  EXPECT_EQ(&graph.GetNode(0), &node);

  // Test InOutNode creation and count
  auto & inOutNode = graph.CreateInOutNode(1, 1);
  EXPECT_EQ(graph.NumNodes(), 2);
  EXPECT_EQ(&graph.GetNode(1), &inOutNode);

  // Test argument node creation and count
  EXPECT_EQ(graph.NumArgumentNodes(), 0);
  auto & argumentNode = graph.CreateArgumentNode();
  EXPECT_EQ(graph.NumArgumentNodes(), 1);
  EXPECT_EQ(&graph.GetArgumentNode(0), &argumentNode);

  // Test result node creation and count
  EXPECT_EQ(graph.NumResultNodes(), 0);
  auto & resultNode = graph.CreateResultNode();
  EXPECT_EQ(graph.NumResultNodes(), 1);
  EXPECT_EQ(&graph.GetResultNode(0), &resultNode);

  // Test finalizing reaching every node
  graph.Finalize();
  EXPECT_TRUE(node.IsFinalized());
  EXPECT_TRUE(argumentNode.IsFinalized());
  EXPECT_TRUE(resultNode.IsFinalized());
  EXPECT_TRUE(inOutNode.IsFinalized());
}

TEST(GraphWriterTests, TestGraphAttributes)
{
  using namespace jlm::util;
  using namespace jlm::util::graph;
  Writer writer;
  auto & graph = writer.CreateGraph();
  graph.SetLabel("My Graph");

  EXPECT_EQ(&graph.GetWriter(), &writer);
  auto & node = graph.CreateNode();

  // Test associating a GraphElement with a pointer, and retrieving it
  int myInt = 6;
  node.SetProgramObject(myInt);
  EXPECT_EQ(&graph.GetFromProgramObject<Node>(myInt), &node);

  // Set some attributes, to test that they appear in the final output
  graph.SetAttributeObject("friend", myInt);
  graph.SetAttributeGraphElement("foe", graph);

  graph.Finalize();

  // Test that the Dot output of the graph contains everything specified
  std::ostringstream out;
  graph.Output(out, OutputFormat::Dot, 0);
  auto string = out.str();

  EXPECT_TRUE(StringContains(string, "label=\"My Graph\""));

  // Nodes referred to in attributes
  EXPECT_TRUE(StringContains(string, "friend=node0"));
  EXPECT_TRUE(StringContains(string, "foe=graph0"));
}

TEST(GraphWriterTests, TestGraphWriterClass)
{
  using namespace jlm::util;
  using namespace jlm::util::graph;
  Writer writer;

  auto & graph0 = writer.CreateGraph();
  auto & graph1 = writer.CreateGraph();
  EXPECT_EQ(writer.NumGraphs(), 2);
  EXPECT_EQ(&writer.GetGraph(0), &graph0);

  auto & node0 = graph0.CreateNode();
  auto & node1 = graph1.CreateNode();

  // Test retrieving a GraphElement from its associated program object pointer
  int myInt = 12;
  node1.SetProgramObject(myInt);
  EXPECT_EQ(writer.GetElementFromProgramObject(reinterpret_cast<uintptr_t>(&myInt)), &node1);

  // Refer to program objects mapped to elements in other graphs
  node0.SetAttributeObject("friend", myInt);

  // Render all the graphs to dot, which first finalizes the graphs to assign unique IDs
  std::ostringstream out;
  writer.outputAllGraphs(out, OutputFormat::Dot);
  auto string = out.str();

  EXPECT_TRUE(graph0.IsFinalized());
  EXPECT_TRUE(graph1.IsFinalized());

  EXPECT_EQ(node0.GetFullId(), "node0");
  EXPECT_EQ(node1.GetFullId(), "node1");

  EXPECT_TRUE(StringContains(string, "friend=node1"));
}

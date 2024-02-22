/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_GRAPHWRITER_HPP
#define JLM_UTIL_GRAPHWRITER_HPP

#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <variant>
#include <vector>

namespace jlm::util
{

enum class GraphOutputFormat
{
  Dot,   // prints
  ASCII, // output format that makes edges implicit when possible
};

enum class AttributeOutputFormat
{
  SpaceSeparatedList, // printed on the form attr=value other="value 2"
  HTMLAttributes      // adds extra restrictions on attribute names
};

class GraphWriter;
class Node;
class Edge;
class Port;
class Graph;

class GraphElement
{
public:
  virtual ~GraphElement() = default;

  /**
   * Constructs a graph element with no label, attributes or associated program object
   * @param label
   */
  GraphElement();

  GraphElement(const GraphElement & other) = delete;

  GraphElement(GraphElement && other) = delete;

  GraphElement &
  operator=(const GraphElement & other) = delete;

  GraphElement &
  operator=(GraphElement && other) = delete;

  /**
   * Gets a short string that will serve as the base for a unique ID.
   * This base should be a valid C-like identifier.
   * @return a string, such as "node", "i", "o", "graph"
   */
  [[nodiscard]] virtual const char *
  GetIdPrefix() const = 0;

  /**
   * Gives the final unique ID of the GraphElement, such as "node3".
   * Requires the GraphElement to be finalized.
   * @return the full id of the GraphElement, including unique suffix
   */
  [[nodiscard]] std::string
  GetFullId() const;

  /**
   * Gets a reference to the graph this GraphElement belongs to
   */
  [[nodiscard]] virtual Graph &
  GetGraph() = 0;

  /**
   * Sets the element's label.
   * A label is text intended to be visible in all renderings of the graph.
   * Use an empty string to signify no label.
   * @param label the new label string
   */
  void
  SetLabel(std::string label);

  /**
   * @return true if this graph element has a non-empty label
   */
  [[nodiscard]] bool
  HasLabel() const;

  /**
   * @return the GraphElement's label
   */
  [[nodiscard]] const std::string &
  GetLabel() const;

  /**
   * @return the graph element's label, or if it is empty, the string \p otherwise
   */
  [[nodiscard]] const char *
  GetLabelOr(const char * otherwise) const;

  /**
   * @return the unique suffix assigned to this element when finalized.
   */
  [[nodiscard]] int
  GetUniqueIdSuffix() const;

  /**
   * Graph elements often represent objects from the program.
   * By making this association explicit, GraphElements can be looked up by program object.
   * When using program objects as attributes, the association is used to refer to
   * the unique id of its associated graph element, instead of the object's address.
   * Within a graph, only one graph element can be associated with any given program object.
   * @param object the object to associate this GraphElement with
   */
  void
  SetProgramObject(void * object)
  {
    SetProgramObjectUintptr(reinterpret_cast<uintptr_t>(object));
  }

  /**
   * @return the program object associated with this graph element.
   */
  [[nodiscard]] uintptr_t
  GetProgramObject() const;

  /**
   * Assigns or overrides a given attribute on the element.
   * @param attribute the name of the attribute
   * @param value the attribute value
   */
  void
  SetAttribute(const std::string & attribute, std::string value);

  /**
   * Assigns or overrides a given attribute on the element with a program object.
   * If this program object is associated with a GraphElement, the attribute value becomes its id.
   * This only works if the element is a part of the same GraphWriter instance.
   * @param attribute the name of the attribute.
   * @param object a pointer to a program object, must be non-null.
   */
  void
  SetAttributeObject(const std::string & attribute, void * object);

  /**
   * Assigns or overrides a given attribute on the element with a reference to a graph element.
   * This allows associations between graph elements to be included in the output, across graphs.
   * The element must be a part of the same GraphWriter instance.
   * @param attribute the name of the attribute.
   * @param element the graph element whose id should be used as attribute value.
   */
  void
  SetAttributeGraphElement(const std::string & attribute, GraphElement & element);

  /**
   * Claims a unique id suffix for the element, if it doesn't already have one.
   */
  virtual void
  Finalize();

  /**
   * @return true if this GraphElement has been finalized, otherwise false.
   */
  [[nodiscard]] bool
  IsFinalized() const;

  /**
   * Prints the attributes of the graph element.
   * @param out the stream output is written to.
   * @param format the output format to use.
   */
  void
  OutputAttributes(std::ostream & out, AttributeOutputFormat format);

private:
  void
  SetProgramObjectUintptr(uintptr_t object);

  // A human-readable piece of text that should be rendered with the element
  std::string Label_;

  // A number added to the end of the id stub to make it globally unique
  int UniqueIdSuffix_;

  // The object in the program this graph object corresponds to, or 0 if none
  uintptr_t ProgramObject_;

  // Arbitrary collection of other attributes. The value can be a string, a reference to a
  // GraphElement, or a reference to a program object.
  using AttributeValue = std::variant<std::string, GraphElement *, uintptr_t>;
  std::unordered_map<std::string, AttributeValue> AttributeMap_;
};

/**
 * Abstract class representing a part of a node that edges can be attached to
 */
class Port : public GraphElement
{
  friend Edge;

protected:
  Port();

public:
  ~Port() override = default;

  virtual Node &
  GetNode() = 0;

  [[nodiscard]] Graph &
  GetGraph() override;

  /**
   * @return true if a directed edge may have its head at this port, false otherwise
   */
  [[nodiscard]] virtual bool
  CanBeEdgeHead() const;

  /**
   * @return true if a directed edge may have its tail at this port, false otherwise
   */
  [[nodiscard]] virtual bool
  CanBeEdgeTail() const;

  /**
   * @return true if any edges are leaving this port, or any non-directed edges are present
   */
  [[nodiscard]] bool
  HasOutgoingEdges() const;

  /**
   * @return true if any edges are pointing at this port, or any non-directed edges are present
   */
  [[nodiscard]] bool
  HasIncomingEdges() const;

  /**
   * Helper function for setting the background color of the Port using the correct dot attributes
   */
  virtual void
  SetFillColor(std::string color) = 0;

  /**
   * Outputs the fully qualified port name, such as node8:i6
   * Only used by the Dot printer.
   */
  virtual void
  OutputDotPortId(std::ostream & out) = 0;

  /**
   * Only used by the ASCII printer.
   * Outputs the origin(s) of edges pointing to this port.
   * Brackets are omitted when exactly one edge points to the port.
   * Example outputs: "o4", "[]" and "[o2, o6]"
   */
  void
  OutputIncomingEdgesASCII(std::ostream & out);

private:
  /**
   * Called when an edge has been added to the graph, to inform the edge's ports
   * @param edge the newly added edge
   */
  void
  OnEdgeAdded(Edge & edge);

  std::vector<Edge *> Connections_;
};

/**
 * Abstract base class for all nodes in the GraphWriter.
 * A Node is also a port, since edges can be attached to nodes directly.
 */
class Node : public Port
{
  friend Graph;

protected:
  explicit Node(Graph & graph);

public:
  ~Node() override = default;

  const char *
  GetIdPrefix() const override;

  Node &
  GetNode() override;

  /**
   * @return the graph this node belongs to
   */
  Graph &
  GetGraph() override;

  void
  SetFillColor(std::string color) override;

  void
  OutputDotPortId(std::ostream & out) override;

  /**
   * Output the node to the ostream \p out, in the specified \p format.
   * Lines printed while outputting are indented by at least \p indent levels.
   * Depending on output format, this function may also recurse and print sub graphs.
   */
  void
  Output(std::ostream & out, GraphOutputFormat format, size_t indent);

  /**
   * Outputs the node in ASCII format to the ostream \p out, indented by \p indent levels.
   * In this format, attributes are ignored, and edges are only included implicitly,
   * by listing the origins of all edges pointing into this node.
   */
  virtual void
  OutputASCII(std::ostream & out, size_t indent);

  /**
   * Outputs the node in Dot format to the ostream \p out, indented by \p indent levels.
   * This format includes all attributes. Edges are output
   */
  virtual void
  OutputDot(std::ostream & out, size_t indent);

  /**
   * Prints all sub graphs of the node, to the given ostream \p out, in the given \p format.
   * * All lines printed by this function are indented by at least \p indent levels.
   * This function is recursive, as sub graphs may have nodes with sub graphs of their own.
   */
  virtual void
  OutputSubgraphs(std::ostream & out, GraphOutputFormat format, size_t indent);

private:
  Graph & Graph_;
};

// InOut nodes represent operations with clearly defined inputs and outputs, with ports for each
class InOutNode;
class InputPort;
class OutputPort;

/**
 * The input port of an InOutNode
 */
class InputPort : public Port
{
  friend InOutNode;

  InputPort(InOutNode & node);

public:
  ~InputPort() override = default;

  const char *
  GetIdPrefix() const override;

  Node &
  GetNode() override;

  bool
  CanBeEdgeTail() const override;

  void
  SetFillColor(std::string color) override;

  void
  OutputDotPortId(std::ostream & out) override;

private:
  InOutNode & Node_;
};

class OutputPort : public Port
{
  friend InOutNode;

  OutputPort(InOutNode & node);

public:
  ~OutputPort() override = default;

  const char *
  GetIdPrefix() const override;

  Node &
  GetNode() override;

  bool
  CanBeEdgeHead() const override;

  void
  SetFillColor(std::string color) override;

  void
  OutputDotPortId(std::ostream & out) override;

private:
  InOutNode & Node_;
};

class InOutNode : public Node
{
  friend Graph;

  InOutNode(Graph & graph, size_t inputPorts, size_t outputPorts);

public:
  ~InOutNode() override = default;

  InputPort &
  CreateInputPort();

  size_t
  NumInputPorts();

  InputPort &
  GetInputPort(size_t index);

  OutputPort &
  CreateOutputPort();

  size_t
  NumOutputPorts();

  OutputPort &
  GetOutputPort(size_t index);

  Graph &
  CreateSubgraph();

  size_t
  NumSubgraphs();

  Graph &
  GetSubgraph(size_t index);

  void
  Finalize() override;

  void
  OutputASCII(std::ostream & out, size_t indent) override;

  void
  OutputDot(std::ostream & out, size_t indent) override;

  void
  OutputSubgraphs(std::ostream & out, GraphOutputFormat format, size_t indent) override;

private:
  std::vector<std::unique_ptr<InputPort>> InputPorts_;
  std::vector<std::unique_ptr<OutputPort>> OutputPorts_;
  std::vector<Graph *> SubGraphs_;
};

/**
 * Node representing a single sink port
 */
class ArgumentNode : public Node
{
  friend Graph;

  ArgumentNode(Graph & graph);

public:
  ~ArgumentNode() override = default;

  const char *
  GetIdPrefix() const override;

  bool
  CanBeEdgeHead() const override;

  /**
   * Indicate that the argument node represents a value coming in from a port in another graph
   * @param outsideSource the Port in the other graph
   */
  void
  SetOutsideSource(Port & outsideSource);

  void
  OutputASCII(std::ostream & out, size_t indent) override;

private:
  // Optional reference to a Port outside of this graph from which this argument came
  Port * OutsideSource_;
};

/**
 * Node representing a single source port
 */
class ResultNode : public Node
{
  friend Graph;

  ResultNode(Graph & graph);

public:
  ~ResultNode() override = default;

  const char *
  GetIdPrefix() const override;

  bool
  CanBeEdgeTail() const override;

  /**
   * Indicate that the result node represents the value of a port in another graph
   * @param outsideSource the Port in the other graph
   */
  void
  SetOutsideDestination(Port & outsideSource);

  void
  OutputASCII(std::ostream & out, size_t indent) override;

private:
  // Optional reference to a Port outside of this graph representing where the result ends up
  Port * OutsideDestination_;
};

class Edge : public GraphElement
{
  friend Graph;

  Edge(Port & from, Port & to, bool directed);

public:
  ~Edge() override = default;

  const char *
  GetIdPrefix() const override;

  Graph &
  GetGraph() override;

  /**
   * Gets the port being pointed to
   * Even if the edge is non-directed, the from/to order can matter for layout.
   */
  [[nodiscard]] Port &
  GetFrom();

  /**
   * Gets the port being pointed from.
   * Even if the edge is non-directed, the from/to order can matter for layout.
   */
  [[nodiscard]] Port &
  GetTo();

  /**
   * @return true if this edge is directed, false otherwise
   */
  [[nodiscard]] bool
  IsDirected() const;

  /**
   * Given one end of the edge, returns the port on the opposite side of the edge.
   */
  [[nodiscard]] Port &
  GetOtherEnd(Port & from);

  /**
   * Outputs the edge in dot format. In ASCII, edges are not implicitly encoded by nodes/ports.
   */
  void
  OutputDot(std::ostream & out, size_t indent);

private:
  Port & From_;
  Port & To_;
  bool Directed_;
};

class Graph : public GraphElement
{
  friend GraphWriter;
  friend GraphElement;

  explicit Graph(GraphWriter & writer);

  Graph(GraphWriter & writer, Node & parentNode);

public:
  ~Graph() override = default;

  const char *
  GetIdPrefix() const override;

  Graph &
  GetGraph() override;

  [[nodiscard]] GraphWriter &
  GetGraphWriter();

  /**
   * @return true if this graph is a subgraph of another graph, false if it is top-level
   */
  [[nodiscard]] bool
  IsSubgraph() const;

  /**
   * Creates a basic Node in the graph. It has a single port: itself.
   * @return a reference to the newly added node.
   */
  [[nodiscard]] Node &
  CreateNode();

  /**
   * Creates an InOutNode in the graph with the given number of input and output ports.
   * @param inputPorts the number of input ports.
   * @param outputPorts the number of output ports.
   * @return a reference to the newly added node.
   */
  [[nodiscard]] InOutNode &
  CreateInOutNode(size_t inputPorts, size_t outputPorts);

  [[nodiscard]] ArgumentNode &
  CreateArgumentNode();

  [[nodiscard]] ResultNode &
  CreateResultNode();

  /**
   * Creates a new edge between from and to. Both ports must belong to this graph.
   * If the edge is directed, the ports must support being the tail and head of an edge.
   * @param from the port the edge goes from.
   * @param to the port the edge goes to.
   * @param directed if true, the edge is a directed edge, otherwise undirected
   * @return a reference to the newly created edge.
   */
  Edge &
  CreateEdge(Port & from, Port & to, bool directed);

  /**
   * Creates a new directed edge from \p from to \p to.
   * @return a reference to the newly created edge.
   * @see CreateEdge
   */
  Edge &
  CreateDirectedEdge(Port & from, Port & to)
  {
    return CreateEdge(from, to, true);
  }

  /**
   * Creates a new edge between \p a and \p b.
   * The ordering of a and b may affect graph layout.
   * @return a reference to the newly created edge.
   * @see CreateEdge
   */
  Edge &
  CreateUndirectedEdge(Port & a, Port & b)
  {
    return CreateEdge(a, b, false);
  }

  /**
   * Retrieves the GraphElement in this graph associated with a given ProgramObject.
   * This function does not look for graph elements inside sub graphs.
   * @param object the program object that is possibly mapped to a GraphElement in the graph
   * @return the GraphElement mapped to the given object, or nullptr if none exists in this graph.
   */
  [[nodiscard]] GraphElement *
  GetElementFromProgramObject(uintptr_t object);

  /**
   * Retrieves the GraphElement in this graph associated with the given program object.
   * Requires the program object to be mapped to a GraphElement in this graph,
   * and that the graph element is of type T.
   * @param object the program object mapped to a GraphElement
   * @return a reference to the T mapped to the given object.
   */
  template<typename T>
  T &
  GetFromProgramObject(void * object)
  {
    static_assert(std::is_base_of_v<GraphElement, T>);
    GraphElement * element = GetElementFromProgramObject(reinterpret_cast<uintptr_t>(object));
    auto result = dynamic_cast<T *>(element);
    JLM_ASSERT(result);
    return *result;
  }

  /**
   * Assigns unique IDs to all graph elements.
   * Finalizing is recursive, visiting all sub graphs.
   */
  void
  Finalize() override;

  /**
   * Prints the graph to the ostream \p out in ASCII.
   * Each line starts with at least \p indent levels of indentation.
   * Requires the graph to be finalized first.
   */
  void
  OutputASCII(std::ostream & out, size_t indent);

  /**
   * Prints the graph to the ostream \p out in dot format.
   * Each line starts with at least \p indent levels of indentation.
   * Requires the graph to be finalized first.
   */
  void
  OutputDot(std::ostream & out, size_t indent);

  /**
   * Prints the graph to the given ostream, in the specified format.
   * Requires the graph to be finalized first.
   * @param out the stream to which output is written
   * @param format the format to output the graph in
   * @param indent the amount of indentation levels the graph should be printed with
   */
  void
  Output(std::ostream & out, GraphOutputFormat format, size_t indent = 0);

private:
  /**
   * Creates a mapping from a GraphElement's assigned program object to the GraphElement.
   * The GraphElement must be a direct member of this graph.
   * @param element the graph element to map
   */
  void
  MapProgramObjectToElement(GraphElement & element);

  /**
   * Removes the mapping of a program object to a graph element in the graph.
   * @param object the program object that should no longer be mapped.
   */
  void
  RemoveProgramObjectMapping(uintptr_t object);

  // The GraphWriter this graph was created by, and belongs to
  GraphWriter & Writer_;

  // If this graph is a subgraph, this is its parent node in the parent graph.
  // For top level graphs, this field is nullptr
  Node * ParentNode_;

  // The set of nodes in the graph. Finalizing the graph may re-order this list.
  std::vector<std::unique_ptr<Node>> Nodes_;

  // Argument nodes and result nodes are kept in separate lists
  std::vector<std::unique_ptr<ArgumentNode>> ArgumentNodes_;
  std::vector<std::unique_ptr<ResultNode>> ResultNodes_;

  std::vector<std::unique_ptr<Edge>> Edges_;

  // A mapping from pointers to program objects, to the GraphElement representing the program object
  std::unordered_map<uintptr_t, GraphElement *> ProgramObjectMapping_;
};

/**
 * Utility class for creating graphs in memory, and printing them to a human or machine readable
 * format.
 */
class GraphWriter
{
public:
  ~GraphWriter() = default;

  GraphWriter() = default;

  GraphWriter(const GraphWriter & other) = delete;

  GraphWriter(GraphWriter && other) = delete;

  GraphWriter &
  operator=(const GraphWriter & other) = delete;

  GraphWriter &
  operator=(GraphWriter && other) = delete;

  [[nodiscard]] Graph &
  CreateGraph();

  /**
   * Attempts to find a GraphElement in one of the graphs that is associated with \p object
   * @return the graph element associated with object, or nullptr if none is found
   */
  [[nodiscard]] GraphElement *
  GetElementFromProgramObject(uintptr_t object);

  /**
   * Finalizes and prints all graphs created in this GraphWriter.
   * @param out the output stream to write graphs to
   * @param format the format to emit the graphs in
   */
  void
  OutputAllGraphs(std::ostream & out, GraphOutputFormat format);

private:
  [[nodiscard]] Graph &
  CreateSubGraph(Node & parentNode);

  friend Graph &
  InOutNode::CreateSubgraph();

  /**
   * Returns a unique suffix for the given \p idStub, starting at 0 and counting up
   * @return the next unique integer suffix for the given idStub
   */
  [[nodiscard]] int
  GetNextUniqueIdStubSuffix(const char * idStub);

  friend void
  GraphElement::Finalize();

  // All graphs being worked on by the GraphWriter
  // Edges can not go across graphs.
  // IDs are however unique across graphs allowing semantic connections.
  std::vector<std::unique_ptr<Graph>> Graphs_;

  // Tracks the next integer to be used when assigning a unique suffix to a given id stub
  std::unordered_map<std::string, int> NextUniqueIdStubSuffix_;
};

}

#endif // JLM_UTIL_GRAPHWRITER_HPP

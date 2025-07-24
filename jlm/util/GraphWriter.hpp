/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_GRAPHWRITER_HPP
#define JLM_UTIL_GRAPHWRITER_HPP

#include <jlm/util/common.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

namespace jlm::util::graph
{

enum class OutputFormat
{
  Dot,   // prints
  ASCII, // output format that makes edges implicit when possible
};

enum class AttributeOutputFormat
{
  SpaceSeparatedList, // printed on the form attr=value other="value 2"
  HTMLAttributes      // adds extra restrictions on attribute names
};

class Writer;
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

  [[nodiscard]] const Graph &
  GetGraph() const;

  /**
   * Sets the element's label.
   * A label is text intended to be visible in all renderings of the graph.
   * Use an empty string to signify no label.
   * @param label the new label string
   */
  void
  SetLabel(std::string label);

  /**
   * Appends the given \p text to the element's label.
   * If the current label is non-empty, the separator string \p sep is inserted between them.
   */
  void
  AppendToLabel(std::string_view text, std::string_view sep = "\n");

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
  [[nodiscard]] std::string_view
  GetLabelOr(std::string_view otherwise) const;

  /**
   * @return the unique suffix assigned to this element when finalized.
   * @see IsFinalized() must return true before calling
   */
  [[nodiscard]] size_t
  GetUniqueIdSuffix() const;

  /**
   * Graph elements often represent objects from the program.
   * By making this association explicit, GraphElements can be looked up by program object.
   * When using program objects as attributes, the association is used to refer to
   * the unique id of its associated graph element, instead of the object's address.
   * Within a graph, only one graph element can be associated with any given program object.
   * @tparam T the type of the program object to associate with
   * @param object the object to associate this GraphElement with
   */
  template<typename T>
  void
  SetProgramObject(const T & object)
  {
    SetProgramObjectUintptr(reinterpret_cast<uintptr_t>(&object));
  }

  /**
   * Removes the association of this GraphElement to any object in the program
   */
  void
  RemoveProgramObject();

  /**
   * @return true, if this GraphElement is associated with any program object
   */
  [[nodiscard]] bool
  HasProgramObject() const noexcept;

  /**
   * @return the program object associated with this graph element.
   */
  [[nodiscard]] uintptr_t
  GetProgramObject() const noexcept;

  /**
   * Assigns or overrides a given attribute on the element.
   * @param attribute the name of the attribute
   * @param value the attribute value
   */
  void
  SetAttribute(const std::string & attribute, std::string value);

  /**
   * Assigns or overrides a given attribute on the element with the address of a program object.
   * If this program object is associated with a GraphElement in the same GraphWriter,
   * the attribute value becomes the id of the other GraphElement, instead of the address.
   * @param attribute the name of the attribute.
   * @param object the address of a program object, must be non-null.
   */
  void
  SetAttributeObject(const std::string & attribute, uintptr_t object);

  /**
   * Helper for calling SetAttributeObject with a pointer to any type
   */
  template<typename T>
  void
  SetAttributeObject(const std::string & attribute, const T & object)
  {
    SetAttributeObject(attribute, reinterpret_cast<uintptr_t>(&object));
  }

  /**
   * Assigns or overrides a given attribute on the element with a reference to a graph element.
   * This allows associations between graph elements to be included in the output, across graphs.
   * The element must be a part of the same GraphWriter instance.
   * @param attribute the name of the attribute.
   * @param element the graph element whose id should be used as attribute value.
   */
  void
  SetAttributeGraphElement(const std::string & attribute, const GraphElement & element);

  /**
   * @return true if an attribute with the given name \p attribute is defined
   */
  [[nodiscard]] bool
  HasAttribute(const std::string & attribute) const;

  /**
   * Retrieves the value of the given \p attribute, if it exists and is assigned a string.
   * @return the attribute's string value, or std::nullopt if it does not exist.
   */
  [[nodiscard]] std::optional<std::string_view>
  GetAttributeString(const std::string & attribute) const;

  /**
   * Retrieves the value of the given \p attribute, if it is assigned a program object.
   * If the attribute does not exist, or is not holding a program object, std::nullopt is returned.
   * @return the object assigned to the attribute, or std::nullopt if it does not exist.
   */
  [[nodiscard]] std::optional<uintptr_t>
  GetAttributeObject(const std::string & attribute) const;

  /**
   * Retrieves the value of the given \p attribute, if it is assigned a graph element.
   * Otherwise, if the attribute is assigned a program object,
   * and there exists a GraphElement representing that program object, that is returned.
   * @return pointer to the GraphElement held in the attribute, or nullptr if it does not exist.
   */
  [[nodiscard]] const GraphElement *
  GetAttributeGraphElement(const std::string & attribute) const;

  /**
   * Removes the attribute with the given name \p attribute, if it exists.
   * @return true if the attribute existed, and was removed, false otherwise
   */
  bool
  RemoveAttribute(const std::string & attribute);

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
  OutputAttributes(std::ostream & out, AttributeOutputFormat format) const;

private:
  void
  SetProgramObjectUintptr(uintptr_t object);

  // A human-readable piece of text that should be rendered with the element
  std::string Label_;

  // A number added to the end of the id stub to make it globally unique
  std::optional<size_t> UniqueIdSuffix_;

  // The object in the program this graph object corresponds to, or 0 if none
  uintptr_t ProgramObject_;

  // Arbitrary collection of other attributes. The value can be a string, a reference to a
  // GraphElement, or a reference to a program object.
  using AttributeValue = std::variant<std::string, const GraphElement *, uintptr_t>;
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

  Graph &
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
   * @return a list of all edges where one end is attached to this port.
   */
  [[nodiscard]] const std::vector<Edge *> &
  GetConnections() const;

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
   * Helper function for setting the background color of the Port using the correct dot attributes.
   * @param color an X11 color name or an RGB value in hex, prefixed by '#'
   * @see jlm::util::Colors namespace for a list of common colors.
   */
  virtual void
  SetFillColor(std::string color) = 0;

  /**
   * Outputs the fully qualified port name, such as node8:i6:n
   * Only used by the Dot printer.
   */
  virtual void
  OutputDotPortId(std::ostream & out) const = 0;

  /**
   * Outputs the origin(s) of edges pointing to this port.
   * Brackets are omitted when exactly one edge points to the port.
   * Only used by the ASCII printer.
   * Example outputs: "o4", "[]" and "[o2, o6]"
   */
  void
  OutputIncomingEdgesASCII(std::ostream & out) const;

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

  /**
   * Sets the shape to be used when rendering the node
   * @see Node::Shape
   */
  virtual void
  SetShape(std::string shape);

  /**
   * A collection of common GraphViz node shapes.
   * See https://graphviz.org/doc/info/shapes.html for more.
   */
  struct Shape
  {
    static inline const char * const Rectangle = "rect";
    static inline const char * const Circle = "circle";
    static inline const char * const Oval = "oval";
    static inline const char * const Point = "point";
    static inline const char * const Plain = "plain";
    static inline const char * const Plaintext = "plaintext";
    static inline const char * const Triangle = "triangle";
    static inline const char * const DoubleCircle = "doublecircle";
  };

  void
  SetFillColor(std::string color) override;

  void
  OutputDotPortId(std::ostream & out) const override;

  /**
   * Output the node to the ostream \p out, in the specified \p format.
   * Lines printed while outputting are indented by at least \p indent levels.
   * Depending on output format, this function may also recurse and print sub graphs.
   */
  void
  Output(std::ostream & out, OutputFormat format, size_t indent) const;

  /**
   * Prints all sub graphs of the node, to the given ostream \p out, in the given \p format.
   * * All lines printed by this function are indented by at least \p indent levels.
   * This function is recursive, as sub graphs may have nodes with sub graphs of their own.
   */
  virtual void
  OutputSubgraphs(std::ostream & out, OutputFormat format, size_t indent) const;

protected:
  /**
   * Outputs the node in ASCII format to the ostream \p out, indented by \p indent levels.
   * In this format, attributes are ignored, and edges are only included implicitly,
   * by listing the origins of all edges pointing into this node.
   */
  virtual void
  OutputASCII(std::ostream & out, size_t indent) const;

  /**
   * Outputs the node in Dot format to the ostream \p out, indented by \p indent levels.
   * This format includes all attributes. Edges are output
   */
  virtual void
  OutputDot(std::ostream & out, size_t indent) const;

private:
  Graph & Graph_;
};

class InOutNode;
class InputPort;
class OutputPort;

/**
 * The input port of an InOutNode
 */
class InputPort final : public Port
{
  friend InOutNode;

  explicit InputPort(InOutNode & node);

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
  OutputDotPortId(std::ostream & out) const override;

private:
  InOutNode & Node_;
};

/**
 * The output port of an InOutNode
 */
class OutputPort final : public Port
{
  friend InOutNode;

  explicit OutputPort(InOutNode & node);

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
  OutputDotPortId(std::ostream & out) const override;

private:
  InOutNode & Node_;
};

/**
 * Class representing a node where data flows into a set of input ports,
 * and results flow out of a set of output ports.
 * For complex operations, the node can also contain one or more sub-graphs.
 */
class InOutNode final : public Node
{
  friend Graph;

  InOutNode(Graph & graph, size_t inputPorts, size_t outputPorts);

public:
  ~InOutNode() override = default;

  /**
   * InOutNodes use HTML tables when rendering, so setting the shape is disabled
   */
  void SetShape(std::string) override;

  InputPort &
  CreateInputPort();

  size_t
  NumInputPorts() const;

  InputPort &
  GetInputPort(size_t index);

  OutputPort &
  CreateOutputPort();

  size_t
  NumOutputPorts() const;

  OutputPort &
  GetOutputPort(size_t index);

  /**
   * Creates a new subgraph and
   * @return a reference to the newly created subgraph
   */
  Graph &
  CreateSubgraph();

  /**
   * @return the number of subgraphs in this node
   */
  size_t
  NumSubgraphs() const;

  /**
   * @return the subgraph with the given \p index, which must be lower than NumSubgraphs()
   */
  Graph &
  GetSubgraph(size_t index);

  /**
   * Set attributes on the HTML-like table used to render the node in dot.
   * See the GraphViz manual's list of table attributes:
   *   https://graphviz.org/doc/info/shapes.html#table
   * @param name the name of the attribute
   * @param value the value the attribute should take
   * @see SetAttribute for setting attributes on the node itself
   */
  void
  SetHtmlTableAttribute(std::string name, std::string value);

  void
  SetFillColor(std::string color) override;

  void
  Finalize() override;

  void
  OutputSubgraphs(std::ostream & out, OutputFormat format, size_t indent) const override;

protected:
  void
  OutputASCII(std::ostream & out, size_t indent) const override;

  void
  OutputDot(std::ostream & out, size_t indent) const override;

private:
  // Attributes that need to be placed on the HTML table in the dot output, and not on the node.
  std::unordered_map<std::string, std::string> HtmlTableAttributes_;

  std::vector<std::unique_ptr<InputPort>> InputPorts_;
  std::vector<std::unique_ptr<OutputPort>> OutputPorts_;
  std::vector<Graph *> SubGraphs_;
};

/**
 * Node representing a port where values enter the graph.
 * All argument nodes are rendered in order at the top of the graph.
 */
class ArgumentNode : public Node
{
  friend Graph;

  explicit ArgumentNode(Graph & graph);

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
  SetOutsideSource(const Port & outsideSource);

protected:
  void
  OutputASCII(std::ostream & out, size_t indent) const override;

private:
  // Optional reference to a Port outside of this graph from which this argument came
  const Port * OutsideSource_;
};

/**
 * Node representing a port where values leave the graph.
 * All result nodes are rendered in order at the bottom of the graph.
 */
class ResultNode : public Node
{
  friend Graph;

  explicit ResultNode(Graph & graph);

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
  SetOutsideDestination(const Port & outsideSource);

protected:
  void
  OutputASCII(std::ostream & out, size_t indent) const override;

private:
  // Optional reference to a Port outside of this graph representing where the result ends up
  const Port * OutsideDestination_;
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
  GetOtherEnd(const Port & end);

  /**
   * Sets the style of the edge
   * @see Edge::Style for a list of possible styles
   */
  void
  SetStyle(std::string style);

  /**
   * The set of available edge styles in GraphViz.
   */
  struct Style
  {
    static inline const char * const Solid = "solid";
    static inline const char * const Dashed = "dashed";
    static inline const char * const Dotted = "dotted";
    static inline const char * const Invisible = "invis";
    static inline const char * const Bold = "bold";
    static inline const char * const Tapered = "tapered";
  };

  /**
   * Customizes the look of the edge at the head end.
   * For a normal arrow, use "normal". Other common options are "box", "diamond" and "dot".
   * Prefix the string with "o" to get outline only. Prefix with "l" or "r" to only get one half.
   * Concatenate multiple strings to get longer arrows, with the tipmost arrow listed first.
   * For full a description of the grammar, see https://graphviz.org/doc/info/arrows.html
   * @param arrow a string describing the look of the edge head.
   */
  void
  SetArrowHead(std::string arrow);

  /**
   * Customizes the look of the edge at the tail end.
   * @param arrow a string describing the look of the edge tail.
   * @see Edge::SetArrowHead() for a short description of the grammar
   */
  void
  SetArrowTail(std::string arrow);

  /**
   * Outputs the edge in dot format. In ASCII, edges are not implicitly encoded by nodes/ports.
   */
  void
  OutputDot(std::ostream & out, size_t indent) const;

private:
  Port & From_;
  Port & To_;
  bool Directed_;
};

class Graph : public GraphElement
{
  friend Writer;
  friend GraphElement;

  explicit Graph(Writer & writer);

  Graph(Writer & writer, Node & parentNode);

public:
  ~Graph() override = default;

  const char *
  GetIdPrefix() const override;

  Graph &
  GetGraph() override;

  [[nodiscard]] Writer &
  GetWriter();

  [[nodiscard]] const Writer &
  GetWriter() const;

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

  /**
   * @return the number of nodes in the graph, excluding argument and result nodes.
   */
  [[nodiscard]] size_t
  NumNodes() const noexcept;

  /**
   * Retrieves the node with the given \p index, which must be lower than NumNodes().
   * Argument nodes and result nodes are not accessed through this function.
   * @return a reference to the node
   */
  [[nodiscard]] Node &
  GetNode(size_t index);

  /**
   * Adds a new argument node to the graph.
   * @return a reference to the new argument node
   */
  [[nodiscard]] ArgumentNode &
  CreateArgumentNode();

  /**
   * @return the number of argument nodes in the graph
   */
  [[nodiscard]] size_t
  NumArgumentNodes() const noexcept;

  /**
   * Retrieves the argument node with the given \p index, which must be less than NumArgumentNodes()
   * @return a reference to the argument node
   */
  [[nodiscard]] Node &
  GetArgumentNode(size_t index);

  /**
   * Adds a new result node to the graph.
   * @return a reference to the new result node
   */
  [[nodiscard]] ResultNode &
  CreateResultNode();

  /**
   * @return the number of result nodes in the graph
   */
  [[nodiscard]] size_t
  NumResultNodes() const noexcept;

  /**
   * Retrieves the result node with the given \p index, which must be less than NumResultNodes()
   * @return a reference to the result node
   */
  [[nodiscard]] Node &
  GetResultNode(size_t index);

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
   * Creates a new undirected edge between \p a and \p b.
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
   * @return the number of edges in the graph
   */
  [[nodiscard]] size_t
  NumEdges() const noexcept;

  /**
   * Retrieves the edge with the given \p index, which must be lower than NumEdges()
   * @return a reference to the edge
   */
  [[nodiscard]] Edge &
  GetEdge(size_t index);

  /**
   * Retrieves an edge connecting ports a and b. If the edge is directed, it must go from a, to b.
   * @param a the first port
   * @param b the second port
   * @return a reference to an edge connecting a and b, or nullptr if no such edge exists.
   */
  [[nodiscard]] Edge *
  GetEdgeBetween(Port & a, Port & b);

  /**
   * Retrieves the GraphElement in this graph associated with a given ProgramObject.
   * This function does not look for graph elements inside sub graphs.
   * @param object the program object that is possibly mapped to a GraphElement in the graph
   * @return the GraphElement mapped to the given object, or nullptr if none exists in this graph.
   */
  [[nodiscard]] GraphElement *
  GetElementFromProgramObject(uintptr_t object) const;

  template<typename T>
  [[nodiscard]] GraphElement *
  GetElementFromProgramObject(const T & object) const
  {
    // Check that object is not a reference to a pointer.
    // If the user truly wants to use the address of a pointer, they can cast it to uintptr_t.
    static_assert(!std::is_pointer_v<T>);
    return GetElementFromProgramObject(reinterpret_cast<uintptr_t>(&object));
  }

  /**
   * Retrieves the GraphElement in this graph associated with the given program object.
   * Requires the program object to be mapped to a GraphElement in this graph,
   * and that the graph element is of type T.
   * @param object the program object mapped to a GraphElement
   * @return a reference to the T mapped to the given object.
   */
  template<typename Element, typename ProgramObject>
  Element &
  GetFromProgramObject(const ProgramObject & object) const
  {
    static_assert(std::is_base_of_v<GraphElement, Element>);
    GraphElement * element = GetElementFromProgramObject(object);
    auto result = dynamic_cast<Element *>(element);
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
   * Prints the graph to the given ostream, in the specified format.
   * Requires the graph to be finalized first.
   * @param out the stream to which output is written
   * @param format the format to output the graph in
   * @param indent the amount of indentation levels the graph should be printed with
   */
  void
  Output(std::ostream & out, OutputFormat format, size_t indent = 0) const;

private:
  void
  OutputASCII(std::ostream & out, size_t indent) const;

  void
  OutputDot(std::ostream & out, size_t indent) const;

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
  Writer & Writer_;

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
class Writer
{
public:
  ~Writer() = default;

  Writer() = default;

  Writer(const Writer & other) = delete;

  Writer(Writer && other) = delete;

  Writer &
  operator=(const Writer & other) = delete;

  Writer &
  operator=(Writer && other) = delete;

  /**
   * Creates a new graph and appends it to the GraphWriter's list of graphs.
   * @return a reference to the newly created graph
   */
  [[nodiscard]] Graph &
  CreateGraph();

  /**
   * @return the number of graphs in the GraphWriter
   */
  [[nodiscard]] size_t
  NumGraphs() const noexcept;

  /**
   * @return a reference to the graph with the given \p index, which must be lower than NumGraphs()
   */
  [[nodiscard]] Graph &
  GetGraph(size_t index);

  /**
   * Attempts to find a GraphElement in one of the graphs that is associated with \p object
   * @return the graph element associated with object, or nullptr if none is found
   */
  [[nodiscard]] GraphElement *
  GetElementFromProgramObject(uintptr_t object) const;

  template<typename T>
  [[nodiscard]] GraphElement *
  GetElementFromProgramObject(const T & object) const
  {
    // Check that object is not a reference to a pointer.
    // If the user truly wants to use the address of a pointer, they can cast it to uintptr_t.
    static_assert(!std::is_pointer_v<T>);
    return GetElementFromProgramObject(reinterpret_cast<uintptr_t>(&object));
  }

  /**
   * Ensures that all graphs added to the graph writer so far are finalized.
   * Recursively finalizes the GraphElements of each graph.
   */
  void
  Finalize();

  /**
   * Finalizes and prints all graphs created in this GraphWriter.
   * @param out the output stream to write graphs to
   * @param format the format to emit the graphs in
   */
  void
  OutputAllGraphs(std::ostream & out, OutputFormat format);

private:
  [[nodiscard]] Graph &
  CreateSubGraph(Node & parentNode);

  friend Graph &
  InOutNode::CreateSubgraph();

  /**
   * Returns a unique suffix for the given \p idStub, starting at 0 and counting up
   * @return the next unique integer suffix for the given idStub
   */
  [[nodiscard]] size_t
  GetNextUniqueIdStubSuffix(const char * idStub);

  friend void
  GraphElement::Finalize();

  // All graphs being worked on by the GraphWriter
  // Edges can not go across graphs.
  // IDs are however unique across graphs allowing semantic connections.
  std::vector<std::unique_ptr<Graph>> Graphs_;

  // Tracks the next integer to be used when assigning a unique suffix to a given id stub
  std::unordered_map<std::string, size_t> NextUniqueIdStubSuffix_;
};

/**
 * List of common color values for use in graph element attributes.
 * You may also use X11 color names or arbitrary hex colors.
 */
namespace Colors
{
inline const char * const Black = "#000000";
inline const char * const Blue = "#0000FF";
inline const char * const Coral = "#FF7F50";
inline const char * const CornflowerBlue = "#6495ED";
inline const char * const Firebrick = " #B22222";
inline const char * const Gold = "#FFD700";
inline const char * const Gray = "#BEBEBE";
inline const char * const Green = "#00FF00";
inline const char * const Orange = "#FFA500";
inline const char * const Purple = "#A020F0";
inline const char * const Red = "#FF0000";
inline const char * const Brown = "#8B4513"; // X11's Saddle Brown
inline const char * const SkyBlue = "#87CEEB";
inline const char * const White = "#FFFFFF";
inline const char * const Yellow = "#FFFF00";
}

}
#endif // JLM_UTIL_GRAPHWRITER_HPP

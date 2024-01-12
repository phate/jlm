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
  Dot,
  ShortASCII, // Only label, no additional attributes
  FullASCII   // Label and all other attributes
};

class GraphWriter;
class Node;
class Edge;
class Port;
class Graph;

class GraphElement
{
protected:
  explicit GraphElement(const std::string & label);

  GraphElement(const GraphElement & other) = delete;
  GraphElement(GraphElement && other) = delete;
  GraphElement &
  operator=(const GraphElement & other) = delete;
  GraphElement &
  operator=(GraphElement && other) = delete;

public:
  virtual ~GraphElement() = default;

  /**
   * Gets a short string that will serve as the base for a unique ID
   * @return a string, such as "node", "i", "o", "graph"
   */
  [[nodiscard]] virtual const char *
  GetIdStub() = 0;

  /**
   * Sets the element's label.
   * A label is intended to be visible in all renderings of the graph
   * @param label the new label string
   */
  void
  SetLabel(const std::string & label);

  /**
   * @return the GraphElement's label
   */
  const std::string &
  GetLabel();

  /**
   * Assigns a suffix to the id stub, unique among all elements sharing id stub, starting at 0
   */
  void
  SetUniqueIdSuffix(int uniqueIdSuffix);

  /**
   * @return the unique suffix assigned to this element
   */
  int
  GetUniqueIdSuffix();

  /**
   * Graph elements often represent objects from the program. By making this association explicit,
   * the GraphElement's unique shorthand id can be found when referencing the object, across graphs.
   */
  void
  SetProgramObject(void * programObject);

  [[nodiscard]] uintptr_t
  GetProgramObject();

  /**
   * Assigns or overrides a given attribute on the element
   * @param attribute the name of the attribute
   * @param value the attribute value
   */
  void
  SetAttributeString(const std::string & attribute, const std::string & value);

  /**
   * Assigns or overrides a given attribute on the element, with a program object.
   * If the object exists somewhere else in the GraphWriter, it will print its id.
   * @param attribute the name of the attribute
   * @param object the program object.
   */
  void
  SetAttributeObject(const std::string & attribute, void * object);

  void
  OutputAttributes(std::ostream & out, GraphOutputFormat format, ) {

  }

  /**
   * Claims a unique id suffix for the element,
   * and registers the ProgramObject with the GraphWriter, if it has one.
   */
  void
  Finalize(GraphWriter & writer);

  /**
   * @return true if this GraphElement has been finalized, false otherwise
   */
  bool IsFinalized();

private:
  // A human-readable piece of text that should be rendered with the element
  std::string Label_;

  // A number added to the end of the id to make it unique, or -1 if still unassigned
  int UniqueIdSuffix_;

  // The object in the program this graph object corresponds to, or 0 if none
  uintptr_t ProgramObject_;

  // Arbitrary collection of other attributes. The value can be a string, or a reference to a
  // program object. In the latter case, the GraphWriter looks for GraphElements representing the
  // other object during printing.
  using AttributeValue = std::variant<std::string, uintptr_t>;
  std::unordered_map<std::string, AttributeValue> AttributeMap_;
};

/**
 * Class representing a part of a node that edges can be attached to
 */
class Port : public GraphElement
{
protected:
  Port(const std::string & label);

public:
  virtual ~Port() = default;

  virtual Node &
  GetNode() = 0;

  void
  OnEdgeAdded(Edge & edge);

protected:
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
  Node(Graph & graph, const std::string & label);

public:
  virtual ~Node() = default;

  virtual const char *
  GetIdStub() override;

  virtual Node &
  GetNode() override;

  /**
   * @return the graph this node belongs to
   */
  Graph &
  GetGraph();

  virtual void
  Output(std::ostream & out, GraphOutputFormat format);

  /**
   * When outputting the nodes of a graph in a sequential fashion, how early should this node be.
   * @return the node's priority, lower values means higher priority, default is 0
   */
  virtual int
  GetOutputOrderPriority();

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
public:
  virtual ~InputPort() = default;

  virtual const char *
  GetIdStub() override;

  virtual Node &
  GetNode() override;

private:
  InOutNode * Node_;
  // The index of this input of the InOutNode
  size_t NodeInputIndex_;
};

class OutputPort : public Port
{
public:
  virtual ~OutputPort() = default;

  virtual const char *
  GetIdStub() override;

  virtual Node &
  GetNode() override;

private:
  InOutNode * Node_;
  // The index of this output of the InOutNode
  size_t NodeOutputIndex_;
};

class InOutNode : public Node
{
  friend Graph;
  InOutNode(Graph & graph, const std::string & label, size_t inPorts, size_t outPorts);

public:
  virtual ~InOutNode() = default;

  size_t
  NumInputPorts();

  InputPort &
  GetInputPort(size_t index);

  size_t
  NumOutputPorts();

  InputPort &
  GetOutputPort(size_t index);

  Graph &
  CreateSubgraph();

  size_t
  NumSubgraphs();

  Graph &
  GetSubgraph(size_t index);

  virtual void
  Output(std::ostream & out, GraphOutputFormat format) override;

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

  ArgumentNode(Graph & graph, const std::string & label);

public:
  /**
   * Indicate that the argument node represents a value coming in from a port in another graph
   * @param outsideSource the Port in the other graph
   */
  void
  SetOutsideSource(Port & outsideSource);

  virtual void
  Output(std::ostream & out, GraphOutputFormat format);

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

  ResultNode(Graph & graph, const std::string & label);

public:
  /**
   * Indicate that the result node represents the value of a port in another graph
   * @param outsideSource the Port in the other graph
   */
  void
  SetOutsideDestination(Port & outsideSource);

  virtual void
  Output(std::ostream & out, GraphOutputFormat format);

private:
  // Optional reference to a Port outside of this graph representing where the result ends up
  Port * OutsideDestination_;
};

class Edge : public GraphElement
{
  Edge(Port * from, Port * to, bool directed);

public:
  virtual ~Edge() = default;

  /**
   * Gets the port being pointed to
   * Even if the edge is non-directed, the from/to order can matter for layout.
   */
  Port &
  GetFrom();

  /**
   * Gets the port being pointed from.
   * Even if the edge is non-directed, the from/to order can matter for layout.
   */
  Port &
  GetTo();

  /**
   * Given one end of the edge, returns the port on the opposite side of the edge.
   */
  Port *
  GetOtherEnd(Port * from);

  virtual void
  Output(std::ostream & out, GraphOutputFormat format);

private:
  Port * From_;
  Port * To_;
  bool Directed_;
};

class Graph : public GraphElement
{
  friend GraphWriter;

  Graph(GraphWriter & writer, const std::string & id);

public:
  Node &
  CreateNode();

  InOutNode &
  CreateInOutNode(size_t inPorts, size_t outPorts);

  ArgumentNode &
  CreateArgumentNode();

  ResultNode &
  CreateResultNode();

  Edge &
  CreateEdge(Port * from, Port * to, bool directed);

  /**
   * Assigns unique ids to all graph elements, and determines the final order of nodes.
   */
  void
  Finalize();

  /**
   * Prints the graph to the given string stream, in the specified format
   * @param ss the stream to which output is written
   * @param format the format to output the graph in
   */
  void
  Output(std::ostream & out, GraphOutputFormat format);

private:
  // The GraphWriter this graph was created by, and belongs to
  GraphWriter & Writer_;

  std::vector<std::unique_ptr<Node>> Nodes_;
  std::vector<std::unique_ptr<Edge>> Edges_;
};

/**
 * Utility class for creating graphs in memory, and printing them to a human or machine readable
 * format
 */
class GraphWriter
{
  friend GraphElement;

public:
  GraphWriter() = default;
  ~GraphWriter() = default;

  Graph &
  CreateGraph(const std::string & label);

  void
  OutputAllGraphs(std::ostream & out, GraphOutputFormat format);

  int
  GetNextUniqueIdStubSuffix(const char * idStub);

  void
  AssociateElementWithProgramObject(GraphElement * elm, uintptr_t programObject);

private:
  // All graphs being worked on by the GraphWriter
  // Edges can not go across graphs, IDs are unique across graphs allowing semantic connections
  std::vector<std::unique_ptr<Graph>> Graphs_;

  // Tracks the next integer to be used when assigning a unique suffix to a given id stub
  std::unordered_map<std::string, int> NextUniqueIdStubSuffix;

  // A shared mapping between objects in the program, and the GraphElements representing them.
  std::unordered_map<uintptr_t, GraphElement *> MappedElements_;
};

/**
 * Returns a shared global instance of the GraphWriter
 */
GraphWriter &
GetGraphWriter();

};

#endif // JLM_GRAPHWRITER_HPP

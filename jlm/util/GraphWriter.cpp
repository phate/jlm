/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/common.hpp>
#include <jlm/util/GraphWriter.hpp>

namespace jlm::util
{

/**
 * Checks if the provided \p string looks like a regular C identifier.
 * The string may only contain alphanumeric characters and underscore, and not start with a digit.
 * @return true if the passed string passes as a C identifier.
 */
bool
LooksLikeIdentifier(std::string_view string)
{
  if (string.empty())
    return false;

  // We avoid C's isalpha, as it is locale dependent
  auto isAlpha = [](char c)
  {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  };
  auto isDigit = [](char c)
  {
    return (c >= '0' && c <= '9');
  };

  char firstChar = string[0];
  if (!isAlpha(firstChar) && firstChar != '_')
    return false;

  for (char c : string)
    if (!isAlpha(c) && !isDigit(c) && c != '_')
      return false;

  return true;
}

/**
 * Prints the given \p string to \p out, while escaping special characters.
 * Unless the string looks like a regular C / Dot identifier, it is surrounded in quotes.
 */
void
PrintIdentifierSafe(std::ostream & out, std::string_view string)
{
  bool quoted = !LooksLikeIdentifier(string);

  if (quoted)
    out << '"';
  for (char c : string)
  {
    if (c == '"')
      out << "\\\"";
    else if (c == '\n')
      out << "\\n";
    else if (c == '\r')
      out << "\\r";
    else if (c == '\t')
      out << "\\t";
    else if (c < ' ' || c >= 127)
    {
      // Print all other special chars as \x escaped hex.
      char tmpStr[3];
      snprintf(tmpStr, sizeof(tmpStr), "%02X", c);
      out << "\\x" << tmpStr;
    }
    else
      out << c;
  }
  if (quoted)
    out << '"';
}

/**
 * Prints the given \p string to \p out with HTML special chars escaped.
 */
void
PrintStringAsHtmlText(std::ostream & out, std::string_view string)
{
  for (char c : string)
  {
    if (c == '&')
      out << "&amp;";
    else if (c == '"')
      out << "&quot;";
    else if (c == '<')
      out << "&lt;";
    else if (c == '>')
      out << "&gt;";
    else if (c == '\n')
      out << "<br>";
    else
      out << c;
  }
}

/**
 * Prints the given \p string to \p out, ignoring any chars that are not in the english alphabet.
 */
void
PrintStringAsHtmlAttributeName(std::ostream & out, std::string_view string)
{
  for (char c : string)
  {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
      out << c;
  }
}

/**
 * Returns a C string with the given amount of indentation.
 * The string is valid until this function is called again.
 * @param indent the number of indentation levels
 * @return a string of spaces corresponding to the indentation level.
 */
[[nodiscard]] const char *
Indent(size_t indent)
{
  static constexpr size_t SPACE_PER_INDENT = 2;
  static constexpr size_t MAX_SPACES = 128;
  static thread_local char indentation[MAX_SPACES + 1];

  size_t spaces = indent * SPACE_PER_INDENT;
  if (spaces > MAX_SPACES)
    spaces = MAX_SPACES;

  for (size_t i = 0; i < spaces; i++)
    indentation[i] = ' ';
  indentation[spaces] = '\0';
  return indentation;
}

GraphElement::GraphElement()
    : Label_(),
      UniqueIdSuffix_(-1),
      ProgramObject_(0),
      AttributeMap_()
{}

std::string
GraphElement::GetFullId() const
{
  JLM_ASSERT(IsFinalized());
  std::ostringstream ss;
  ss << GetIdStub() << GetUniqueIdSuffix();
  return ss.str();
}

void
GraphElement::SetLabel(std::string label)
{
  Label_ = std::move(label);
}

bool
GraphElement::HasLabel() const
{
  return !Label_.empty();
}

const std::string &
GraphElement::GetLabel() const
{
  return Label_;
}

const char *
GraphElement::GetLabelOr(const char * otherwise) const
{
  if (HasLabel())
    return Label_.c_str();
  return otherwise;
}

int
GraphElement::GetUniqueIdSuffix() const
{
  return UniqueIdSuffix_;
}

void
GraphElement::SetProgramObjectUintptr(uintptr_t object)
{
  JLM_ASSERT(object);
  if (ProgramObject_ != 0)
    GetGraph().RemoveProgramObjectMapping(ProgramObject_);
  ProgramObject_ = object;
  GetGraph().MapProgramObjectToElement(*this);
}

uintptr_t
GraphElement::GetProgramObject() const
{
  return ProgramObject_;
}

void
GraphElement::SetAttributeString(const std::string & attribute, const std::string & value)
{
  AttributeMap_[attribute] = value;
}

void
GraphElement::SetAttributeUintptr(const std::string & attribute, uintptr_t object)
{
  JLM_ASSERT(object);
  AttributeMap_[attribute] = object;
}

void
GraphElement::Finalize()
{
  if (IsFinalized())
    return;

  auto & writer = GetGraph().GetGraphWriter();
  UniqueIdSuffix_ = writer.GetNextUniqueIdStubSuffix(GetIdStub());
}

bool
GraphElement::IsFinalized() const
{
  return UniqueIdSuffix_ != -1;
}

void
GraphElement::OutputAttributes(std::ostream & out, AttributeOutputFormat format)
{
  for (const auto & [name, value] : AttributeMap_)
  {
    if (format == AttributeOutputFormat::SpaceSeparatedList)
      PrintIdentifierSafe(out, name);
    else if (format == AttributeOutputFormat::HTMLAttributes)
      PrintStringAsHtmlAttributeName(out, name);
    else
      JLM_UNREACHABLE("Unknown AttributeOutputFormat");

    out << "=";
    if (auto * string = std::get_if<std::string>(&value))
    {
      if (format == AttributeOutputFormat::SpaceSeparatedList)
        PrintIdentifierSafe(out, *string);
      else
      {
        out << '"';
        PrintStringAsHtmlText(out, *string);
        out << '"';
      }
    }
    else if (auto * ptr = std::get_if<uintptr_t>(&value))
    {
      // Check if some GraphElement in the graph is mapped to this pointer
      auto * element = GetGraph().GetElementFromProgramObject(*ptr);

      // If not, check all other graphs
      if (element == nullptr)
        element = GetGraph().GetGraphWriter().GetElementFromProgramObject(*ptr);

      if (element)
        out << element->GetFullId();
      else
        out << "ptr" << element;
    }
    out << " ";
  }
}

Port::Port()
    : GraphElement()
{}

Graph &
Port::GetGraph()
{
  return GetNode().GetGraph();
}

bool
Port::CanBeEdgeHead() const
{
  return true;
}

bool
Port::CanBeEdgeTail() const
{
  return true;
}

void
Port::OnEdgeAdded(jlm::util::Edge & edge)
{
  if (this == &edge.GetFrom())
    JLM_ASSERT(CanBeEdgeTail() || !edge.IsDirected());
  else if (this == &edge.GetTo())
    JLM_ASSERT(CanBeEdgeHead() || !edge.IsDirected());
  else
    JLM_UNREACHABLE("Port was informed about unrelated edge");

  Connections_.push_back(&edge);
}

bool
Port::HasOutgoingEdges() const
{
  for (auto & edge : Connections_)
  {
    if (&edge->GetFrom() == this || !edge->IsDirected())
      return true;
  }
  return false;
}

bool
Port::HasIncomingEdges() const
{
  for (auto & edge : Connections_)
  {
    if (&edge->GetTo() == this || !edge->IsDirected())
      return true;
  }
  return false;
}

void
Port::OutputIncomingEdgesASCII(std::ostream & out)
{
  std::ostringstream text;
  size_t numIncomingEdges = 0;

  for (auto & edge : Connections_)
  {
    if (&edge->GetTo() != this && edge->IsDirected())
      continue;

    Port & otherEnd = edge->GetOtherEnd(*this);
    if (numIncomingEdges == 0)
      text << otherEnd.GetFullId();
    else
      text << ", " << otherEnd.GetFullId();

    numIncomingEdges++;
  }

  if (numIncomingEdges == 1)
    out << text.rdbuf();
  else
    out << "[" << text.rdbuf() << "]";
}

Node::Node(Graph & graph)
    : Port(),
      Graph_(graph)
{}

const char *
Node::GetIdStub() const
{
  return "node";
}

Node &
Node::GetNode()
{
  return *this;
}

Graph &
Node::GetGraph()
{
  return Graph_;
}

void
Node::OutputDotPortId(std::ostream & out)
{
  out << GetFullId();
}

void
Node::Output(std::ostream & out, GraphOutputFormat format, size_t indent)
{
  JLM_ASSERT(IsFinalized());

  switch (format)
  {
  case GraphOutputFormat::ASCII:
  {
    out << Indent(indent);
    if (HasOutgoingEdges())
    {
      out << GetFullId() << ":";
    }
    out << GetLabelOr("NODE");
    if (HasIncomingEdges())
    {
      out << "<-";
      OutputIncomingEdgesASCII(out);
    }
    break;
  }
  case GraphOutputFormat::Dot:
  {
    out << Indent(indent) << GetFullId() << " [";
    out << "label=";
    if (HasLabel())
    {
      PrintIdentifierSafe(out, GetLabel());
      out << " tooltip=";
      PrintIdentifierSafe(out, GetFullId());
    }
    else
    {
      PrintIdentifierSafe(out, GetFullId());
    }
    out << " ";
    OutputAttributes(out, AttributeOutputFormat::SpaceSeparatedList);
    out << "]" << std::endl;
    break;
  }
  default:
    JLM_UNREACHABLE("Unknown GraphOutputFormat");
  }
}

void
Node::OutputSubgraphs(std::ostream & out, GraphOutputFormat format, size_t indent)
{
  // Regular nodes do not have sub graphs
}

InputPort::InputPort(jlm::util::InOutNode & node, size_t nodeInputIndex)
    : Node_(node),
      NodeInputIndex_(nodeInputIndex)
{}

const char *
InputPort::GetIdStub() const
{
  return "i";
}

Node &
InputPort::GetNode()
{
  return Node_;
}

bool
InputPort::CanBeEdgeTail() const
{
  return false;
}

void
InputPort::OutputDotPortId(std::ostream & out)
{
  out << Node_.GetFullId() << ":" << GetFullId();
}

OutputPort::OutputPort(jlm::util::InOutNode & node, size_t nodeOutputIndex)
    : Node_(node),
      NodeOutputIndex_(nodeOutputIndex)
{}

const char *
OutputPort::GetIdStub() const
{
  return "o";
}

Node &
OutputPort::GetNode()
{
  return Node_;
}

bool
OutputPort::CanBeEdgeHead() const
{
  return false;
}

void
OutputPort::OutputDotPortId(std::ostream & out)
{
  out << Node_.GetFullId() << ":" << GetFullId();
}

InOutNode::InOutNode(Graph & graph, size_t inputPorts, size_t outputPorts)
    : Node(graph)
{
  for (size_t i = 0; i < inputPorts; i++)
    CreateInputPort();

  for (size_t i = 0; i < outputPorts; i++)
    CreateOutputPort();
}

InputPort &
InOutNode::CreateInputPort()
{
  auto * inputPort = new InputPort(*this, InputPorts_.size());
  InputPorts_.emplace_back(inputPort);
  return *inputPort;
}

size_t
InOutNode::NumInputPorts()
{
  return InputPorts_.size();
}

InputPort &
InOutNode::GetInputPort(size_t index)
{
  JLM_ASSERT(index < InputPorts_.size());
  return *InputPorts_[index];
}

OutputPort &
InOutNode::CreateOutputPort()
{
  auto * outputPort = new OutputPort(*this, OutputPorts_.size());
  OutputPorts_.emplace_back(outputPort);
  return *outputPort;
}

size_t
InOutNode::NumOutputPorts()
{
  return InputPorts_.size();
}

OutputPort &
InOutNode::GetOutputPort(size_t index)
{
  JLM_ASSERT(index < OutputPorts_.size());
  return *OutputPorts_[index];
}

Graph &
InOutNode::CreateSubgraph()
{
  auto & graph = GetGraph().GetGraphWriter().CreateSubGraph(*this);
  SubGraphs_.push_back(&graph);
  return graph;
}

size_t
InOutNode::NumSubgraphs()
{
  return SubGraphs_.size();
}

Graph &
InOutNode::GetSubgraph(size_t index)
{
  JLM_ASSERT(index < SubGraphs_.size());
  return *SubGraphs_[index];
}

void
InOutNode::Finalize()
{
  Node::Finalize();

  for (auto & inputPort : InputPorts_)
    inputPort->Finalize();
  for (auto & outputPort : OutputPorts_)
    outputPort->Finalize();
  for (auto & graph : SubGraphs_)
    graph->Finalize();
}

void
InOutNode::Output(std::ostream & out, GraphOutputFormat format, size_t indent)
{
  switch (format)
  {
  case GraphOutputFormat::ASCII:
  {
    out << Indent(indent);

    // output the names of all output ports
    for (size_t i = 0; i < NumOutputPorts(); i++)
    {
      if (i != 0)
        out << ", ";
      out << GetOutputPort(i).GetFullId();
    }
    if (NumOutputPorts() != 0)
      out << " := ";

    // If the node itself is used as a tail port, we must include its name
    if (Port::HasOutgoingEdges())
    {
      out << GetFullId() << ":";
    }
    out << GetLabelOr("NODE");
    if (Port::HasIncomingEdges())
    {
      out << "<-";
      Port::OutputIncomingEdgesASCII(out);
    }
    out << " ";

    // Now output the origins of all input ports
    for (size_t i = 0; i < NumInputPorts(); i++)
    {
      if (i != 0)
        out << ", ";
      GetInputPort(i).OutputIncomingEdgesASCII(out);
    }
    out << std::endl;

    // Output all sub graphs, if we have any
    OutputSubgraphs(out, format, indent + 1);

    break;
  }
  case GraphOutputFormat::Dot:
  {
    out << Indent(indent) << GetFullId() << " [shape=plain ";
    out << "label=<" << std::endl;

    // InOutNodes are printed as html tables
    out << "<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='0' CELLPADDING='0'>" << std::endl;

    // Inputs
    if (!InputPorts_.empty())
    {
      out << "<TR>" << std::endl;
      for (auto & inputPort : InputPorts_)
      {
        out << "<TD PORT='" << inputPort->GetFullId() << "' ";
        inputPort->OutputAttributes(out, AttributeOutputFormat::HTMLAttributes);
        out << ">";
        PrintStringAsHtmlText(out, inputPort->GetLabelOr(inputPort->GetFullId().c_str()));
        out << "</TD>" << std::endl;;
      }
      out << "</TR>" << std::endl;
    }

    // Label
    out << "<TR><TD>" << std::endl;
    PrintStringAsHtmlText(out, GetLabelOr(GetFullId().c_str()));
    out << "</TD></TR>" << std::endl;

    // Subgraphs
    if (!SubGraphs_.empty())
    {
      out << "<TR>" << std::endl;
      for (auto & graph : SubGraphs_)
      {
        out << "<TD SUBGRAPH='" << graph->GetFullId() << "'>";
        PrintStringAsHtmlText(out, graph->GetLabelOr(graph->GetFullId().c_str()));
        out << "</TD>" << std::endl;;
      }
      out << "</TR>" << std::endl;
    }

    // Outputs
    if (!OutputPorts_.empty())
    {
      out << "<TR>" << std::endl;
      for (auto & outputPort : OutputPorts_)
      {
        out << "<TD PORT='" << outputPort->GetFullId() << "' ";
        outputPort->OutputAttributes(out, AttributeOutputFormat::HTMLAttributes);
        out << ">";
        PrintStringAsHtmlText(out, outputPort->GetLabelOr(outputPort->GetFullId().c_str()));
        out << "</TD>" << std::endl;
      }
      out << "</TR>" << std::endl;
    }

    out << "</TABLE>" << std::endl;
    out << Indent(indent) << "> ";
    OutputAttributes(out, AttributeOutputFormat::SpaceSeparatedList);
    out << "]" << std::endl;
    break;
  }
  default:
    JLM_UNREACHABLE("Unknown GraphOutputFormat");
  }
}

void
InOutNode::OutputSubgraphs(std::ostream & out, GraphOutputFormat format, size_t indent)
{
  for (auto & graph : SubGraphs_)
    graph->Output(out, format, indent);
}

ArgumentNode::ArgumentNode(jlm::util::Graph & graph)
    : Node(graph),
      OutsideSource_(nullptr)
{}

const char *
ArgumentNode::GetIdStub() const
{
  return "a";
}

bool
ArgumentNode::CanBeEdgeHead() const
{
  return false;
}

void
ArgumentNode::SetOutsideSource(Port & outsideSource)
{
  OutsideSource_ = &outsideSource;
}

void
ArgumentNode::Output(std::ostream & out, jlm::util::GraphOutputFormat format, size_t indent)
{
  switch (format)
  {
  case GraphOutputFormat::ASCII:
  {
    // In ASCII the argument is printed as part of an ARG line
    out << GetFullId();
    if (OutsideSource_ != nullptr)
      out << " <= " << OutsideSource_->GetFullId();
    break;
  }
  case GraphOutputFormat::Dot:
  {
    Node::Output(out, format, indent);
    break;
  }
  default:
    JLM_UNREACHABLE("Unknown GraphOutputFormat");
  }
}

ResultNode::ResultNode(jlm::util::Graph & graph)
    : Node(graph),
      OutsideDestination_(nullptr)
{}

const char *
ResultNode::GetIdStub() const
{
  return "r";
}

bool
ResultNode::CanBeEdgeTail() const
{
  return false;
}

void
ResultNode::SetOutsideDestination(Port & outsideDestination)
{
  OutsideDestination_ = &outsideDestination;
}

void
ResultNode::Output(std::ostream & out, jlm::util::GraphOutputFormat format, size_t indent)
{
  switch (format)
  {
  case GraphOutputFormat::ASCII:
  {
    // In ASCII the result is printed as part of an RES line
    out << GetFullId();
    if (OutsideDestination_ != nullptr)
      out << " => " << OutsideDestination_->GetFullId();
    break;
  }
  case GraphOutputFormat::Dot:
  {
    Node::Output(out, format, indent);
    break;
  }
  default:
    JLM_UNREACHABLE("Unknown GraphOutputFormat");
  }
}

Edge::Edge(Port & from, Port & to, bool directed)
    : From_(from),
      To_(to),
      Directed_(directed)
{
  from.OnEdgeAdded(*this);
  to.OnEdgeAdded(*this);
}

const char *
Edge::GetIdStub() const
{
  return "edge";
}

Graph &
Edge::GetGraph()
{
  // from and to have the same graph, return either
  return From_.GetGraph();
}

Port &
Edge::GetFrom()
{
  return From_;
}

Port &
Edge::GetTo()
{
  return To_;
}

bool
Edge::IsDirected() const
{
  return Directed_;
}

Port &
Edge::GetOtherEnd(Port & from)
{
  if (&from == &From_)
    return To_;
  else if (&from == &To_)
    return From_;

  JLM_UNREACHABLE("GetOtherEnd called with neither end");
}

void
Edge::Output(std::ostream & out, GraphOutputFormat format, size_t indent)
{
  switch (format)
  {
  case GraphOutputFormat::ASCII:
    break; // All edge information is described in terms of inputs and outputs
  case GraphOutputFormat::Dot:
  {
    out << Indent(indent);
    From_.OutputDotPortId(out);
    out << " -> ";
    To_.OutputDotPortId(out);
    out << "[";
    if (!Directed_)
      out << "dir=none;";
    if (HasLabel())
    {
      out << "label=";
      PrintIdentifierSafe(out, GetLabel());
      out << " ";
    }
    OutputAttributes(out, AttributeOutputFormat::SpaceSeparatedList);
    out << "]";
    break;
  }
  default:
    JLM_UNREACHABLE("Unknown GraphOutputFormat");
  }
}

Graph::Graph(GraphWriter & writer)
    : GraphElement(),
      Writer_(writer),
      ParentNode_(nullptr)
{}

Graph::Graph(GraphWriter & writer, Node & parentNode)
    : GraphElement(),
      Writer_(writer),
      ParentNode_(&parentNode)
{}

const char *
Graph::GetIdStub() const
{
  return "graph";
}

Graph &
Graph::GetGraph()
{
  return *this;
}

GraphWriter &
Graph::GetGraphWriter()
{
  return Writer_;
}

bool
Graph::IsSubgraph() const
{
  return ParentNode_ != nullptr;
}

Node &
Graph::CreateNode()
{
  auto * node = new Node(*this);
  Nodes_.emplace_back(std::unique_ptr<Node>(node));
  return *node;
}

InOutNode &
Graph::CreateInOutNode(size_t inputPorts, size_t outputPorts)
{
  auto * node = new InOutNode(*this, inputPorts, outputPorts);
  Nodes_.emplace_back(std::unique_ptr<InOutNode>(node));
  return *node;
}

ArgumentNode &
Graph::CreateArgumentNode()
{
  auto * node = new ArgumentNode(*this);
  ArgumentNodes_.emplace_back(std::unique_ptr<ArgumentNode>(node));
  return *node;
}

ResultNode &
Graph::CreateResultNode()
{
  auto * node = new ResultNode(*this);
  ResultNodes_.emplace_back(std::unique_ptr<ResultNode>(node));
  return *node;
}

Edge &
Graph::CreateEdge(Port & from, Port & to, bool directed)
{
  // Edges must be between ports in this graph
  JLM_ASSERT(&from.GetGraph() == this);
  JLM_ASSERT(&to.GetGraph() == this);

  // Edge's constructor informs the ports about the edge
  auto * edge = new Edge(from, to, directed);
  Edges_.emplace_back(std::unique_ptr<Edge>(edge));
  return *edge;
}

GraphElement *
Graph::GetElementFromProgramObject(uintptr_t object)
{
  auto it = ProgramObjectMapping_.find(object);
  if (it != ProgramObjectMapping_.end())
    return it->second;
  return nullptr;
}

void
Graph::MapProgramObjectToElement(GraphElement & element)
{
  JLM_ASSERT(&element.GetGraph() == this);

  uintptr_t object = element.GetProgramObject();
  JLM_ASSERT(object != 0);

  auto & slot = ProgramObjectMapping_[object];
  JLM_ASSERT(slot == nullptr && "Trying to map a GraphElement to an already mapped program object");
  slot = &element;
}

void
Graph::RemoveProgramObjectMapping(uintptr_t object)
{
  size_t erased = ProgramObjectMapping_.erase(object);
  JLM_ASSERT(erased == 1);
}

void
Graph::Finalize()
{
  GraphElement::Finalize();

  for (auto & arg : ArgumentNodes_)
    arg->Finalize();
  // Nodes with sub graphs also finalize them
  for (auto & node : Nodes_)
    node->Finalize();
  for (auto & res : ResultNodes_)
    res->Finalize();
  for (auto & edge : Edges_)
    edge->Finalize();
}

void
Graph::Output(std::ostream & out, jlm::util::GraphOutputFormat format, size_t indent)
{
  JLM_ASSERT(IsFinalized());

  switch (format)
  {
  case GraphOutputFormat::ASCII:
  {
    out << Indent(indent) << "{" << std::endl;
    indent++;

    // Use a single ARG line for all graph arguments
    bool anyArguments = false;
    for (auto & arg : ArgumentNodes_)
    {
      if (!anyArguments)
        out << Indent(indent) << "ARG ";
      else
        out << ", ";
      anyArguments = true;
      arg->Output(out, format, indent);
    }
    if (anyArguments)
      out << std::endl;

    // Print all other nodes in order
    for (auto & node : Nodes_)
    {
      // Will also print sub graphs recursively
      node->Output(out, format, indent);
    }

    // Use a single RES line for all graph results
    bool anyResults = false;
    for (auto & arg : ArgumentNodes_)
    {
      if (!anyResults)
        out << Indent(indent) << "RES ";
      else
        out << ", ";
      anyResults = true;
      arg->Output(out, format, indent);
    }
    if (anyResults)
      out << std::endl;

    indent--;
    out << Indent(indent) << "}" << std::endl;

    break;
  }
  case GraphOutputFormat::Dot:
  {
    out << Indent(indent) << "digraph " << GetFullId() << " {" << std::endl;
    indent++;

    out << Indent(indent) << "node[shape=box];" << std::endl;
    out << Indent(indent) << "penwidth=6;" << std::endl;
    for (auto & node : Nodes_)
    {
      node->Output(out, format, indent);
    }
    for (auto & edge : Edges_)
    {
      edge->Output(out, format, indent);
    }

    indent--;
    out << Indent(indent) << "}" << std::endl;

    // After fully printing this graph, print any sub graphs it may have
    for (auto & node : Nodes_)
    {
      node->OutputSubgraphs(out, format, indent);
    }
    break;
  }
  default:
    JLM_UNREACHABLE("Unknown output format");
  }
}

Graph &
GraphWriter::CreateGraph()
{
  auto * graph = new Graph(*this);
  Graphs_.emplace_back(std::unique_ptr<Graph>(graph));
  return *graph;
}

Graph &
GraphWriter::CreateSubGraph(Node & parentNode)
{
  auto * graph = new Graph(*this, parentNode);
  Graphs_.emplace_back(std::unique_ptr<Graph>(graph));
  return *graph;
}

GraphElement *
GraphWriter::GetElementFromProgramObject(uintptr_t object)
{
  for (auto & graph : Graphs_)
    if (auto * found = graph->GetElementFromProgramObject(object))
      return found;

  return nullptr;
}

int
GraphWriter::GetNextUniqueIdStubSuffix(const char * idStub)
{
  int & nextValue = NextUniqueIdStubSuffix_[idStub];
  return nextValue++;
}

void
GraphWriter::OutputAllGraphs(std::ostream & out, GraphOutputFormat format)
{
  for (auto & graph : Graphs_)
    if (!graph->IsSubgraph())
      graph->Finalize();

  for (auto & graph : Graphs_)
    if (!graph->IsSubgraph())
      graph->Output(out, format);
}

GraphWriter &
GetGraphWriter()
{
  static GraphWriter theGraphWriter;
  return theGraphWriter;
}

}

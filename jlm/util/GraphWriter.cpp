/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/GraphWriter.hpp>

#include <jlm/util/strfmt.hpp>

namespace jlm::util::graph
{
// All GraphElements with an associated ProgramObject get this attribute added
static const char * const TOOLTIP_ATTRIBUTE = "tooltip";
// Edges are not named in dot, so use an attribute to assign id instead
static const char * const EDGE_ID_ATTRIBUTE = "id";

/**
 * Checks if the provided \p string looks like a regular C identifier.
 * The string may only contain alphanumeric characters and underscore, and not start with a digit.
 * @return true if the passed string passes as a C identifier.
 */
static bool
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
static void
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
 * If \p replaceNewlines is true, newlines are replaced by <BR/>, otherwise they are kept as is.
 * Newlines are allowed inside HTML attributes, but are ignored in HTML text.
 */
static void
PrintStringAsHtmlText(std::ostream & out, std::string_view string, bool replaceNewlines)
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
    else if (c == '\n' && replaceNewlines)
      out << "<BR/>";
    else
      out << c;
  }
}

/**
 * Prints the given \p string to \p out,
 * replacing chars that are not allowed in html attribute names by '-'
 */
static void
PrintStringAsHtmlAttributeName(std::ostream & out, std::string_view string)
{
  for (char c : string)
  {
    if (c <= ' ' || c >= 127 || c == '<' || c == '>' || c == '"' || c == '\'' || c == '/'
        || c == '=')
      out << '-';
    else
      out << c;
  }
}

/**
 * Returns a C string with the given amount of indentation.
 * The string is valid until this function is called again.
 * @param indent the number of indentation levels
 * @return a string of spaces corresponding to the indentation level.
 */
[[nodiscard]] static const char *
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
      UniqueIdSuffix_(std::nullopt),
      ProgramObject_(0),
      AttributeMap_()
{}

std::string
GraphElement::GetFullId() const
{
  JLM_ASSERT(IsFinalized());
  std::ostringstream ss;
  ss << GetIdPrefix() << GetUniqueIdSuffix();
  return ss.str();
}

const Graph &
GraphElement::GetGraph() const
{
  return const_cast<GraphElement *>(this)->GetGraph();
}

void
GraphElement::SetLabel(std::string label)
{
  Label_ = std::move(label);
}

void
GraphElement::AppendToLabel(std::string_view text, std::string_view sep)
{
  if (HasLabel())
  {
    Label_.append(sep).append(text);
  }
  else
  {
    Label_ = text;
  }
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

std::string_view
GraphElement::GetLabelOr(std::string_view otherwise) const
{
  if (HasLabel())
    return Label_;
  return otherwise;
}

size_t
GraphElement::GetUniqueIdSuffix() const
{
  JLM_ASSERT(UniqueIdSuffix_);
  return UniqueIdSuffix_.value();
}

void
GraphElement::SetProgramObjectUintptr(uintptr_t object)
{
  JLM_ASSERT(object);
  if (ProgramObject_ != 0)
    GetGraph().RemoveProgramObjectMapping(ProgramObject_);
  ProgramObject_ = object;
  if (ProgramObject_ != 0)
    GetGraph().MapProgramObjectToElement(*this);
}

void
GraphElement::RemoveProgramObject()
{
  SetProgramObjectUintptr(0);
}

bool
GraphElement::HasProgramObject() const noexcept
{
  return ProgramObject_ != 0;
}

uintptr_t
GraphElement::GetProgramObject() const noexcept
{
  return ProgramObject_;
}

void
GraphElement::SetAttribute(const std::string & attribute, std::string value)
{
  AttributeMap_[attribute] = std::move(value);
}

void
GraphElement::SetAttributeObject(const std::string & attribute, uintptr_t object)
{
  JLM_ASSERT(object);
  AttributeMap_[attribute] = object;
}

void
GraphElement::SetAttributeGraphElement(const std::string & attribute, const GraphElement & element)
{
  JLM_ASSERT(&GetGraph().GetWriter() == &element.GetGraph().GetWriter());
  AttributeMap_[attribute] = &element;
}

bool
GraphElement::HasAttribute(const std::string & attribute) const
{
  return AttributeMap_.find(attribute) != AttributeMap_.end();
}

std::optional<std::string_view>
GraphElement::GetAttributeString(const std::string & attribute) const
{
  if (auto it = AttributeMap_.find(attribute); it != AttributeMap_.end())
  {
    if (auto stringValue = std::get_if<std::string>(&it->second))
    {
      return *stringValue;
    }
  }
  return std::nullopt;
}

std::optional<uintptr_t>
GraphElement::GetAttributeObject(const std::string & attribute) const
{
  if (auto it = AttributeMap_.find(attribute); it != AttributeMap_.end())
  {
    if (auto uintptrValue = std::get_if<uintptr_t>(&it->second))
    {
      return *uintptrValue;
    }
  }
  return std::nullopt;
}

const GraphElement *
GraphElement::GetAttributeGraphElement(const std::string & attribute) const
{
  if (auto it = AttributeMap_.find(attribute); it != AttributeMap_.end())
  {
    if (auto graphElementValue = std::get_if<const GraphElement *>(&it->second))
    {
      return *graphElementValue;
    }

    // Otherwise, check if this attribute holds a program object that is represented by a
    // GraphElement in this graph, or in any graph in the GraphWriter.
    if (auto ptr = std::get_if<uintptr_t>(&it->second))
    {
      if (auto gElement = GetGraph().GetElementFromProgramObject(*ptr))
      {
        return gElement;
      }
      if (auto gwElement = GetGraph().GetWriter().GetElementFromProgramObject(*ptr))
      {
        return gwElement;
      }
    }
  }
  return nullptr;
}

bool
GraphElement::RemoveAttribute(const std::string & attribute)
{
  return AttributeMap_.erase(attribute);
}

void
GraphElement::Finalize()
{
  if (IsFinalized())
    return;

  auto & writer = GetGraph().GetWriter();
  UniqueIdSuffix_ = writer.GetNextUniqueIdStubSuffix(GetIdPrefix());
}

bool
GraphElement::IsFinalized() const
{
  return UniqueIdSuffix_.has_value();
}

void
GraphElement::OutputAttributes(std::ostream & out, AttributeOutputFormat format) const
{
  auto FormatAttribute = [&](std::string_view name, std::string_view value)
  {
    if (format == AttributeOutputFormat::SpaceSeparatedList)
    {
      PrintIdentifierSafe(out, name);
      out << "=";
      PrintIdentifierSafe(out, value);
    }
    else if (format == AttributeOutputFormat::HTMLAttributes)
    {
      PrintStringAsHtmlAttributeName(out, name);
      out << "=\""; // HTML attributes must be quoted
      PrintStringAsHtmlText(out, value, false);
      out << '"'; // Closing quote
    }
    else
    {
      JLM_UNREACHABLE("Unknown AttributeOutputFormat");
    }

    out << " "; // Attributes are space separated in both formats
  };

  auto OutputAttribute = [&](const std::string & name)
  {
    if (auto string = GetAttributeString(name))
    {
      FormatAttribute(name, *string);
    }
    else if (auto graphElement = GetAttributeGraphElement(name))
    {
      FormatAttribute(name, graphElement->GetFullId());
    }
    else if (auto object = GetAttributeObject(name))
    {
      FormatAttribute(name, strfmt("ptr", std::hex, *object));
    }
    else
    {
      JLM_UNREACHABLE("Unknown attribute type");
    }
  };

  for (const auto & [name, _] : AttributeMap_)
  {
    OutputAttribute(name);
  }

  // If no other tooltip is set, print the address of the associated program object
  if (HasProgramObject() && !HasAttribute(TOOLTIP_ATTRIBUTE))
    FormatAttribute(TOOLTIP_ATTRIBUTE, strfmt(std::hex, GetProgramObject()));
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

const std::vector<Edge *> &
Port::GetConnections() const
{
  return Connections_;
}

void
Port::OnEdgeAdded(Edge & edge)
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
Port::OutputIncomingEdgesASCII(std::ostream & out) const
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
    out << text.str();
  else
    out << "[" << text.str() << "]";
}

Node::Node(Graph & graph)
    : Port(),
      Graph_(graph)
{}

const char *
Node::GetIdPrefix() const
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
Node::SetShape(std::string shape)
{
  SetAttribute("shape", std::move(shape));
}

void
Node::SetFillColor(std::string color)
{
  // The dot output gives all nodes style=filled by default, so we only need to set the color
  SetAttribute("fillcolor", std::move(color));
}

void
Node::OutputDotPortId(std::ostream & out) const
{
  out << GetFullId();
}

void
Node::Output(std::ostream & out, OutputFormat format, size_t indent) const
{
  switch (format)
  {
  case OutputFormat::ASCII:
    OutputASCII(out, indent);
    break;
  case OutputFormat::Dot:
    OutputDot(out, indent);
    break;
  default:
    JLM_UNREACHABLE("Unknown graph::OutputFormat");
  }
}

void
Node::OutputASCII(std::ostream & out, size_t indent) const
{
  out << Indent(indent);
  if (HasOutgoingEdges())
  {
    out << GetFullId() << ":";
  }
  PrintIdentifierSafe(out, GetLabelOr("NODE"));
  if (HasIncomingEdges())
  {
    out << "<-";
    OutputIncomingEdgesASCII(out);
  }
  out << std::endl;
}

void
Node::OutputDot(std::ostream & out, size_t indent) const
{
  out << Indent(indent) << GetFullId() << " [";
  out << "label=";
  if (HasLabel())
  {
    PrintIdentifierSafe(out, GetLabel());
  }
  else
  {
    PrintIdentifierSafe(out, GetFullId());
  }
  out << " ";
  OutputAttributes(out, AttributeOutputFormat::SpaceSeparatedList);
  out << "];" << std::endl;
}

void
Node::OutputSubgraphs(std::ostream & out, OutputFormat format, size_t indent) const
{
  // Regular nodes do not have sub graphs
}

InputPort::InputPort(InOutNode & node)
    : Node_(node)
{
  SetFillColor(Colors::White);
}

const char *
InputPort::GetIdPrefix() const
{
  return "in";
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
InputPort::SetFillColor(std::string color)
{
  // Attribute on the <TD> tag used by the dot output
  SetAttribute("BGCOLOR", std::move(color));
}

void
InputPort::OutputDotPortId(std::ostream & out) const
{
  out << Node_.GetFullId() << ":" << GetFullId() << ":n";
}

OutputPort::OutputPort(InOutNode & node)
    : Node_(node)
{
  SetFillColor(Colors::White);
}

const char *
OutputPort::GetIdPrefix() const
{
  return "out";
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
OutputPort::SetFillColor(std::string color)
{
  // Attribute on the <TD> tag used by the dot output
  SetAttribute("BGCOLOR", std::move(color));
}

void
OutputPort::OutputDotPortId(std::ostream & out) const
{
  out << Node_.GetFullId() << ":" << GetFullId() << ":s";
}

InOutNode::InOutNode(Graph & graph, size_t inputPorts, size_t outputPorts)
    : Node(graph)
{
  for (size_t i = 0; i < inputPorts; i++)
    CreateInputPort();

  for (size_t i = 0; i < outputPorts; i++)
    CreateOutputPort();

  SetFillColor(Colors::White);
}

void
InOutNode::SetShape(std::string)
{
  throw Error("InOutNodes can not have custom shapes set");
}

InputPort &
InOutNode::CreateInputPort()
{
  auto inputPort = new InputPort(*this);
  InputPorts_.emplace_back(inputPort);
  return *inputPort;
}

size_t
InOutNode::NumInputPorts() const
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
  auto outputPort = new OutputPort(*this);
  OutputPorts_.emplace_back(outputPort);
  return *outputPort;
}

size_t
InOutNode::NumOutputPorts() const
{
  return OutputPorts_.size();
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
  auto & graph = GetGraph().GetWriter().CreateSubGraph(*this);
  SubGraphs_.push_back(&graph);
  return graph;
}

size_t
InOutNode::NumSubgraphs() const
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
InOutNode::SetHtmlTableAttribute(std::string name, std::string value)
{
  HtmlTableAttributes_[name] = std::move(value);
}

void
InOutNode::SetFillColor(std::string color)
{
  SetHtmlTableAttribute("BGCOLOR", std::move(color));
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
InOutNode::OutputSubgraphs(std::ostream & out, OutputFormat format, size_t indent) const
{
  for (auto & graph : SubGraphs_)
    graph->Output(out, format, indent);
}

void
InOutNode::OutputASCII(std::ostream & out, size_t indent) const
{
  out << Indent(indent);

  // output the names of all output ports
  for (size_t i = 0; i < NumOutputPorts(); i++)
  {
    if (i != 0)
      out << ", ";
    out << OutputPorts_[i]->GetFullId();
  }
  if (NumOutputPorts() != 0)
    out << " := ";

  // If the node itself is used as a tail port, we must include its name
  if (Port::HasOutgoingEdges())
  {
    out << GetFullId() << ":";
  }
  PrintIdentifierSafe(out, GetLabelOr("NODE"));
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
    InputPorts_[i]->OutputIncomingEdgesASCII(out);
  }
  out << std::endl;

  // Output all sub graphs, if we have any
  OutputSubgraphs(out, OutputFormat::ASCII, indent + 1);
}

void
InOutNode::OutputDot(std::ostream & out, size_t indent) const
{
  out << Indent(indent) << GetFullId() << " [shape=plain style=solid ";
  out << "label=<" << std::endl;

  // InOutNodes are printed as html tables
  out << "<TABLE BORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\">" << std::endl;

  // Used to create rows of boxes above and below the node
  auto PrintPortList = [&out](auto & ports)
  {
    if (ports.empty())
      return;

    out << "\t<TR><TD>" << std::endl;
    out << "\t\t<TABLE BORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\"><TR>" << std::endl;
    out << "\t\t\t<TD WIDTH=\"20\"></TD>" << std::endl;
    for (size_t i = 0; i < ports.size(); i++)
    {
      // Spacing
      if (i != 0)
        out << "\t\t\t<TD WIDTH=\"10\"></TD>" << std::endl;

      auto & port = *ports[i];
      out << "\t\t\t<TD BORDER=\"1\" CELLPADDING=\"1\" ";
      out << "PORT=\"" << port.GetFullId() << "\" ";
      port.OutputAttributes(out, AttributeOutputFormat::HTMLAttributes);
      if (port.HasLabel())
      {
        out << "><FONT POINT-SIZE=\"10\">";
        PrintStringAsHtmlText(out, port.GetLabel(), true);
        out << "</FONT>";
      }
      else
      {
        // ports without labels have a fixed size
        out << " WIDTH=\"8\" HEIGHT=\"5\" FIXEDSIZE=\"true\">";
      }
      out << "</TD>" << std::endl;
    }
    out << "\t\t\t<TD WIDTH=\"20\"></TD>" << std::endl;
    out << "\t\t</TR></TABLE>" << std::endl;
    out << "\t</TD></TR>" << std::endl;
  };

  // Inputs
  PrintPortList(InputPorts_);

  // The main body of the node: a rounded rectangle
  out << "\t<TR><TD>" << std::endl;
  out << "\t\t<TABLE BORDER=\"1\" STYLE=\"ROUNDED\" CELLBORDER=\"0\" ";
  out << "CELLSPACING=\"0\" CELLPADDING=\"0\" ";
  for (auto & [name, value] : HtmlTableAttributes_)
  {
    PrintStringAsHtmlAttributeName(out, name);
    out << "=\"";
    PrintStringAsHtmlText(out, value, false);
    out << "\" ";
  }
  out << ">" << std::endl;
  out << "\t\t\t<TR><TD CELLPADDING=\"1\">";
  PrintStringAsHtmlText(out, GetLabelOr(GetFullId()), true);
  out << "</TD></TR>" << std::endl;

  // Subgraphs
  if (!SubGraphs_.empty())
  {
    out << "\t\t\t<TR><TD>" << std::endl;
    out << "\t\t\t\t<TABLE BORDER=\"0\" CELLSPACING=\"4\" CELLPADDING=\"2\"><TR>" << std::endl;
    for (auto & graph : SubGraphs_)
    {
      out << "\t\t\t\t\t<TD BORDER=\"1\" STYLE=\"ROUNDED\" WIDTH=\"40\" BGCOLOR=\"white\" ";
      out << "_SUBGRAPH=\"" << graph->GetFullId() << "\">";
      PrintStringAsHtmlText(out, graph->GetFullId(), true);
      out << "</TD>" << std::endl;
    }
    out << "\t\t\t\t</TR></TABLE>" << std::endl;
    out << "\t\t\t</TD></TR>" << std::endl;
  }

  // End of the rounded rectangle
  out << "\t\t</TABLE>" << std::endl;
  out << "\t</TD></TR>" << std::endl;

  PrintPortList(OutputPorts_);

  out << "</TABLE>" << std::endl;
  out << Indent(indent) << "> ";
  OutputAttributes(out, AttributeOutputFormat::SpaceSeparatedList);
  out << "];" << std::endl;
}

ArgumentNode::ArgumentNode(Graph & graph)
    : Node(graph),
      OutsideSource_(nullptr)
{}

const char *
ArgumentNode::GetIdPrefix() const
{
  return "arg";
}

bool
ArgumentNode::CanBeEdgeHead() const
{
  return false;
}

void
ArgumentNode::SetOutsideSource(const Port & outsideSource)
{
  OutsideSource_ = &outsideSource;
  SetAttributeGraphElement("outsideSource", outsideSource);
}

void
ArgumentNode::OutputASCII(std::ostream & out, size_t) const
{
  // In ASCII the argument is printed as part of an ARG line
  out << GetFullId();
  if (HasLabel())
  {
    out << ":";
    PrintIdentifierSafe(out, GetLabel());
  }
  if (OutsideSource_ != nullptr)
  {
    out << " <= ";
    OutsideSource_->OutputIncomingEdgesASCII(out);
  }
}

ResultNode::ResultNode(Graph & graph)
    : Node(graph),
      OutsideDestination_(nullptr)
{}

const char *
ResultNode::GetIdPrefix() const
{
  return "res";
}

bool
ResultNode::CanBeEdgeTail() const
{
  return false;
}

void
ResultNode::SetOutsideDestination(const Port & outsideDestination)
{
  OutsideDestination_ = &outsideDestination;
  SetAttributeGraphElement("outsideDest", outsideDestination);
}

void
ResultNode::OutputASCII(std::ostream & out, size_t) const
{
  // In ASCII the result is printed as part of an RES line
  OutputIncomingEdgesASCII(out);
  if (HasLabel())
  {
    out << ":";
    PrintIdentifierSafe(out, GetLabel());
  }
  if (OutsideDestination_ != nullptr)
    out << " => " << OutsideDestination_->GetFullId();
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
Edge::GetIdPrefix() const
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
Edge::GetOtherEnd(const Port & end)
{
  if (&end == &From_)
    return To_;
  else if (&end == &To_)
    return From_;

  JLM_UNREACHABLE("GetOtherEnd called with neither end");
}

void
Edge::SetStyle(std::string style)
{
  SetAttribute("style", std::move(style));
}

void
Edge::SetArrowHead(std::string arrow)
{
  SetAttribute("arrowhead", std::move(arrow));
}

void
Edge::SetArrowTail(std::string arrow)
{
  // When outputting dot, the "dir" attribute will be automatically changed to make the tail visible
  SetAttribute("arrowtail", std::move(arrow));
}

void
Edge::OutputDot(std::ostream & out, size_t indent) const
{
  out << Indent(indent);
  From_.OutputDotPortId(out);
  out << " -> ";
  To_.OutputDotPortId(out);
  out << "[";

  const bool hasHeadArrow = HasAttribute("arrowhead") || Directed_;
  const bool hasTailArrow = HasAttribute("arrowtail");
  if (hasHeadArrow && hasTailArrow)
    out << "dir=both ";
  else if (hasHeadArrow)
  {
    // dir=forward is the default in digraphs
  }
  else if (hasTailArrow)
    out << "dir=back ";
  else
    out << "dir=none ";

  if (HasLabel())
  {
    out << "label=";
    PrintIdentifierSafe(out, GetLabel());
    out << " ";
  }

  // Edges are not normally named, so use the id attribute to include the edge's id
  if (!HasAttribute(EDGE_ID_ATTRIBUTE))
  {
    out << EDGE_ID_ATTRIBUTE << "=";
    PrintIdentifierSafe(out, GetFullId());
    out << " ";
  }

  OutputAttributes(out, AttributeOutputFormat::SpaceSeparatedList);
  out << "];" << std::endl;
}

Graph::Graph(Writer & writer)
    : GraphElement(),
      Writer_(writer),
      ParentNode_(nullptr)
{}

Graph::Graph(Writer & writer, Node & parentNode)
    : GraphElement(),
      Writer_(writer),
      ParentNode_(&parentNode)
{}

const char *
Graph::GetIdPrefix() const
{
  return "graph";
}

Graph &
Graph::GetGraph()
{
  return *this;
}

Writer &
Graph::GetWriter()
{
  return Writer_;
}

const Writer &
Graph::GetWriter() const
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
  auto node = new Node(*this);
  Nodes_.emplace_back(node);
  return *node;
}

InOutNode &
Graph::CreateInOutNode(size_t inputPorts, size_t outputPorts)
{
  auto node = new InOutNode(*this, inputPorts, outputPorts);
  Nodes_.emplace_back(node);
  return *node;
}

size_t
Graph::NumNodes() const noexcept
{
  return Nodes_.size();
}

Node &
Graph::GetNode(size_t index)
{
  JLM_ASSERT(index < NumNodes());
  return *Nodes_[index];
}

ArgumentNode &
Graph::CreateArgumentNode()
{
  auto node = new ArgumentNode(*this);
  ArgumentNodes_.emplace_back(node);
  return *node;
}

size_t
Graph::NumArgumentNodes() const noexcept
{
  return ArgumentNodes_.size();
}

Node &
Graph::GetArgumentNode(size_t index)
{
  JLM_ASSERT(index < NumArgumentNodes());
  return *ArgumentNodes_[index];
}

ResultNode &
Graph::CreateResultNode()
{
  auto node = new ResultNode(*this);
  ResultNodes_.emplace_back(node);
  return *node;
}

size_t
Graph::NumResultNodes() const noexcept
{
  return ResultNodes_.size();
}

Node &
Graph::GetResultNode(size_t index)
{
  JLM_ASSERT(index < NumResultNodes());
  return *ResultNodes_[index];
}

Edge &
Graph::CreateEdge(Port & from, Port & to, bool directed)
{
  // Edges must be between ports in this graph
  JLM_ASSERT(&from.GetGraph() == this);
  JLM_ASSERT(&to.GetGraph() == this);

  // Edge's constructor informs the ports about the edge
  auto edge = new Edge(from, to, directed);
  Edges_.emplace_back(edge);
  return *edge;
}

size_t
Graph::NumEdges() const noexcept
{
  return Edges_.size();
}

Edge &
Graph::GetEdge(size_t index)
{
  JLM_ASSERT(index < NumEdges());
  return *Edges_[index];
}

Edge *
Graph::GetEdgeBetween(Port & a, Port & b)
{
  for (auto edge : a.GetConnections())
  {
    if (edge->IsDirected() && &edge->GetFrom() != &a)
      continue;
    if (&edge->GetOtherEnd(a) == &b)
      return edge;
  }
  return nullptr;
}

GraphElement *
Graph::GetElementFromProgramObject(uintptr_t object) const
{
  if (auto it = ProgramObjectMapping_.find(object); it != ProgramObjectMapping_.end())
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
Graph::OutputASCII(std::ostream & out, size_t indent) const
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
    arg->OutputASCII(out, indent);
  }
  if (anyArguments)
    out << std::endl;

  // Print all other nodes in order
  for (auto & node : Nodes_)
  {
    // Will also print sub graphs recursively
    node->OutputASCII(out, indent);
  }

  // Use a single RES line for all graph results
  bool anyResults = false;
  for (auto & res : ResultNodes_)
  {
    if (!anyResults)
      out << Indent(indent) << "RES ";
    else
      out << ", ";
    anyResults = true;
    res->OutputASCII(out, indent);
  }
  if (anyResults)
    out << std::endl;

  indent--;
  out << Indent(indent) << "}" << std::endl;
}

void
Graph::OutputDot(std::ostream & out, size_t indent) const
{
  out << Indent(indent) << "digraph " << GetFullId() << " {" << std::endl;
  indent++;

  // Default node attributes. Filling nodes by default makes them easier to click
  out << Indent(indent)
      << "node[shape=box style=filled fillcolor=white width=0.1 height=0.1 margin=0.05];"
      << std::endl;
  out << Indent(indent) << "penwidth=6;" << std::endl;
  if (HasLabel())
  {
    out << Indent(indent) << "label=";
    PrintIdentifierSafe(out, GetLabel());
    out << std::endl;
  }
  out << Indent(indent);
  OutputAttributes(out, AttributeOutputFormat::SpaceSeparatedList);
  out << std::endl;

  // Helper function used to print argument nodes and result nodes
  auto PrintOrderedSubgraph = [&out](auto & nodes, const char * rank, size_t indent)
  {
    if (nodes.empty())
      return;
    out << Indent(indent++) << "{" << std::endl;
    out << Indent(indent) << "rank=" << rank << ";" << std::endl;
    for (size_t i = 0; i < nodes.size(); i++)
    {
      nodes[i]->OutputDot(out, indent);

      // Use invisible edges to order nodes in the subgraph
      if (i != 0)
        out << Indent(indent) << nodes[i - 1]->GetFullId() << " -> " << nodes[i]->GetFullId()
            << "[style=invis];" << std::endl;
    }
    out << Indent(--indent) << "}" << std::endl;
  };

  PrintOrderedSubgraph(ArgumentNodes_, "source", indent);

  for (auto & node : Nodes_)
  {
    node->OutputDot(out, indent);
  }

  PrintOrderedSubgraph(ResultNodes_, "sink", indent);

  for (auto & edge : Edges_)
  {
    edge->OutputDot(out, indent);
  }

  indent--;
  out << Indent(indent) << "}" << std::endl;

  // After fully printing this graph, print any sub graphs it may have
  for (auto & node : Nodes_)
  {
    node->OutputSubgraphs(out, OutputFormat::Dot, indent);
  }
}

void
Graph::Output(std::ostream & out, OutputFormat format, size_t indent) const
{
  JLM_ASSERT(IsFinalized());

  switch (format)
  {
  case OutputFormat::ASCII:
    OutputASCII(out, indent);
    break;
  case OutputFormat::Dot:
    OutputDot(out, indent);
    break;
  default:
    JLM_UNREACHABLE("Unknown output format");
  }
}

Graph &
Writer::CreateGraph()
{
  auto graph = new Graph(*this);
  Graphs_.emplace_back(graph);
  return *graph;
}

size_t
Writer::NumGraphs() const noexcept
{
  return Graphs_.size();
}

Graph &
Writer::GetGraph(size_t index)
{
  JLM_ASSERT(index < NumGraphs());
  return *Graphs_[index];
}

Graph &
Writer::CreateSubGraph(Node & parentNode)
{
  auto graph = new Graph(*this, parentNode);
  Graphs_.emplace_back(graph);
  return *graph;
}

GraphElement *
Writer::GetElementFromProgramObject(uintptr_t object) const
{
  for (auto & graph : Graphs_)
    if (auto found = graph->GetElementFromProgramObject(object))
      return found;

  return nullptr;
}

size_t
Writer::GetNextUniqueIdStubSuffix(const char * idStub)
{
  size_t & nextValue = NextUniqueIdStubSuffix_[idStub];
  return nextValue++;
}

void
Writer::Finalize()
{
  for (auto & graph : Graphs_)
    if (!graph->IsSubgraph())
      graph->Finalize();
}

void
Writer::OutputAllGraphs(std::ostream & out, OutputFormat format)
{
  Finalize();

  for (auto & graph : Graphs_)
    if (!graph->IsSubgraph())
      graph->Output(out, format);
}

}

/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/common.hpp>
#include <jlm/util/GraphWriter.hpp>

namespace jlm::util
{

/**
 * Checks if the provided \p string looks like an identifier,
 * and doesn't need to be surrounded in double quotes.
 * Also checks against dot keywords, as defined at https://graphviz.org/doc/info/lang.html
 * @return true if the passed string is formatted as an identifier, false otherwise.
 */
bool
IsDotIdentifier(std::string_view string)
{
  if (string.size() == 0)
    return false;

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
  {
    if (!isAlpha(c) && !isDigit(c) && c != '_')
      return false;
  }

  // TODO: Check against dot keywords

  return true;
}

/**
 * Prints the given \p string to \p out.
 * If the string is not a valid identifier,
 */
void
PrintIdentifierSafe(std::ostream & out, std::string_view string)
{
  std::stringstream escaped;

  bool hasSpecialCharacters = false;

  for (char c : string)
  {
    if (c == '"')
      escaped << "\\\"";
    else if (c == '\n')
      escaped << "\\n";
    else if (c == '\r')
      escaped << "\\r";
    else if (c == '\t')
      escaped << "\\t";
    else if (c < ' ')
    {
      char tmpStr[3];
      snprintf(tmpStr, sizeof tmpStr, "%02X", c);
      escaped << "\\x" << tmpStr;
    }
    else
      escaped << c;
  }
}

/**
 * Prints the given \p string to \p out with HTML special chars escaped
 */
void
PrintStringAsHtmlText(std::ostream & out, std::string_view string)
{
  for (char c : string)
  {
    if (c == '&')
      out << "&amp;";
    if (c == '"')
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

GraphElement::GraphElement(const std::string & label)
    : Label_(label),
      UniqueIdSuffix_(-1),
      ProgramObject_(0),
      AttributeMap_()
{}

void
GraphElement::SetLabel(const std::string & label)
{
  Label_ = label;
}

const std::string &
GraphElement::GetLabel()
{
  return Label_;
}

void
GraphElement::SetUniqueIdSuffix(int uniqueIdSuffix)
{
  UniqueIdSuffix_ = uniqueIdSuffix;
}

int
GraphElement::GetUniqueIdSuffix()
{
  return UniqueIdSuffix_;
}

void
GraphElement::SetProgramObject(void * programObject)
{
  ProgramObject_ = reinterpret_cast<uintptr_t>(programObject);
}

uintptr_t
GraphElement::GetProgramObject()
{
  return ProgramObject_;
}

void
GraphElement::SetAttributeString(const std::string & attribute, const std::string & value)
{
  AttributeMap_[attribute] = value;
}

void
GraphElement::SetAttributeObject(const std::string & attribute, void * object)
{
  AttributeMap_[attribute] = reinterpret_cast<uintptr_t>(object);
}

void
GraphElement::Finalize(GraphWriter & writer)
{
  if (IsFinalized())
    return;

  UniqueIdSuffix_ = writer.GetNextUniqueIdStubSuffix(GetIdStub());
  if (ProgramObject_ != 0)
    writer.AssociateElementWithProgramObject(this, ProgramObject_);
}

bool
GraphElement::IsFinalized()
{
  return UniqueIdSuffix_ != -1;
}

Port::Port(const std::string & label)
    : GraphElement(label)
{}

void
Port::OnEdgeAdded(jlm::util::Edge & edge)
{
  JLM_ASSERT(this == &edge.GetFrom() || this == &edge.GetTo());
  Connections_.push_back(&edge);
}

Node::Node(Graph & graph, const std::string & label)
    : Port(label),
      Graph_(graph)
{}

const char *
Node::GetIdStub()
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
Node::Output(std::ostream & out, jlm::util::GraphOutputFormat format)
{
  JLM_ASSERT(IsFinalized());

  switch (format)
  {
  case GraphOutputFormat::ShortASCII:
  case GraphOutputFormat::FullASCII:
    out << GetIdStub() << GetUniqueIdSuffix() << '[';
    PrintStringAsQuote(out, GetLabel());
    if (format == GraphOutputFormat::FullASCII)
    {
      OutputAttributes(out, format);
    }
    out << ']' << std::endl;
    return;
  case GraphOutputFormat::Dot:
    return;
  default:
    JLM_UNREACHABLE("Unknown GraphOutputFormat");
  }
}

Graph::Graph(GraphWriter & writer, const std::string & label)
    : GraphElement(label),
      Writer_(writer)
{}

Graph &
GraphWriter::CreateGraph(const std::string & label)
{
  Graph * graph = new Graph(*this, label);
  Graphs_.push_back(std::unique_ptr<Graph>(graph));
  return *graph;
}

void
GraphWriter::Output(std::stringstream & ss)
{
  for (auto & graph : Graphs_)
    graph->ToASCII(ss);
}

GraphWriter &
GetGraphWriter()
{
  static GraphWriter theGraphWriter;
  return theGraphWriter;
}

}

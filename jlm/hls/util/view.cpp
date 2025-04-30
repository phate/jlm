/*
 * Copyright 2024 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/ir/hls.hpp>
#include <jlm/hls/util/view.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/view.hpp>

#include <algorithm>

namespace jlm::hls
{

std::string
ViewcolorToString(const ViewColors & color)
{
  switch (color)
  {
  case NONE:
    return "";
    break;

  case BLACK:
    return "black";
    break;

  case RED:
    return "red";
    break;

  default:
    JLM_UNREACHABLE("HLS view color not defined");
    break;
  }
}

static inline std::string
hex(size_t i)
{
  std::stringstream stream;
  stream << std::hex << i;
  return stream.str();
}

std::string
GetDotName(rvsdg::Node * node)
{
  return util::strfmt("n", hex((intptr_t)node));
}

std::string
GetDotName(rvsdg::output * output)
{
  if (dynamic_cast<rvsdg::RegionArgument *>(output))
  {
    return util::strfmt("a", hex((intptr_t)output), ":", "default");
  }

  if (auto no = dynamic_cast<rvsdg::SimpleOutput *>(output))
  {
    return util::strfmt(GetDotName(no->node()), ":", "o", hex((intptr_t)output));
  }
  else if (dynamic_cast<rvsdg::StructuralOutput *>(output))
  {
    return util::strfmt("so", hex((intptr_t)output), ":", "default");
  }
  JLM_UNREACHABLE("not implemented");
}

template<class T>
ViewColors
GetDefaultColor(std::unordered_map<T *, ViewColors> & map, T * elem, ViewColors def = BLACK)
{
  auto f = map.find(elem);
  if (f == map.end())
  {
    return def;
  }
  return f->second;
}

template<class T>
ViewColors
GetDefaultLabel(std::unordered_map<T *, ViewColors> & map, T * elem, ViewColors def = NONE)
{
  auto f = map.find(elem);
  if (f == map.end())
  {
    return def;
  }
  return f->second;
}

std::string
GetDotName(rvsdg::input * input)
{
  if (dynamic_cast<rvsdg::RegionResult *>(input))
  {
    return util::strfmt("r", hex((intptr_t)input), ":", "default");
  }
  if (auto ni = dynamic_cast<rvsdg::SimpleInput *>(input))
  {
    return util::strfmt(GetDotName(ni->node()), ":", "i", hex((intptr_t)input));
  }
  else if (dynamic_cast<rvsdg::StructuralInput *>(input))
  {
    return util::strfmt("si", hex((intptr_t)input), ":", "default");
  }
  JLM_UNREACHABLE("not implemented");
}

std::string
PortToDot(const std::string & display_name, const std::string & dot_name, const ViewColors & color)
{
  auto dot =
      dot_name
      + " [shape=plaintext label=<\n"
        "            <TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\">\n"
        "                <TR>\n"
        "                    <TD PORT=\"default\" BORDER=\"1\" CELLPADDING=\"1\"><FONT "
        "POINT-SIZE=\"10\">"
      + display_name
      + "</FONT></TD>\n"
        "                </TR>\n"
        "            </TABLE>\n"
        "> tooltip=\""
      + dot_name + "\" color=" + ViewcolorToString(color) + " fontcolor=" + ViewcolorToString(color)
      + "];\n";
  return dot;
}

std::string
ArgumentToDot(rvsdg::RegionArgument * argument, const ViewColors & color)
{
  auto display_name = util::strfmt("a", argument->index());
  auto dot_name = util::strfmt("a", hex((intptr_t)argument));
  return PortToDot(display_name, dot_name, color);
}

std::string
ResultToDot(rvsdg::RegionResult * result, const ViewColors & color)
{
  auto display_name = util::strfmt("r", result->index());
  auto dot_name = util::strfmt("r", hex((intptr_t)result));
  return PortToDot(display_name, dot_name, color);
}

std::string
StructuralInputToDot(rvsdg::StructuralInput * structuralInput, const ViewColors & color)
{
  auto display_name = util::strfmt("si", structuralInput->index());
  auto dot_name = util::strfmt("si", hex((intptr_t)structuralInput));
  return PortToDot(display_name, dot_name, color);
}

std::string
StructuralOutputToDot(rvsdg::StructuralOutput * structuralOutput, const ViewColors & color)
{
  auto display_name = util::strfmt("so", structuralOutput->index());
  auto dot_name = util::strfmt("so", hex((intptr_t)structuralOutput));
  return PortToDot(display_name, dot_name, color);
}

std::string
Edge(
    rvsdg::output * output,
    rvsdg::input * input,
    std::unordered_map<rvsdg::output *, ViewColors> & tailLabel,
    bool back_edge = false)
{
  auto color = "black";
  auto tailLabelColor = GetDefaultLabel(tailLabel, output);
  if (!back_edge)
  {
    return GetDotName(output) + " -> " + GetDotName(input)
         + " [style=\"\", arrowhead=\"normal\", color=" + color
         + ", headlabel=<>, fontsize=15, labelangle=45, labeldistance=2.0, labelfontcolor=blue, "
           "tooltip=\""
         + output->type().debug_string() + "\", taillabel=\"" + ViewcolorToString(tailLabelColor)
         + "\"];\n";
  }
  return GetDotName(input) + " -> " + GetDotName(output)
       + " [style=\"\", arrowhead=\"normal\", color=" + color
       + ", headlabel=<>, fontsize=15, labelangle=45, labeldistance=2.0, labelfontcolor=blue, "
         "constraint=false, tooltip=\""
       + output->type().debug_string() + "\", taillabel=\"" + ViewcolorToString(tailLabelColor)
       + "\"];\n";
}

std::string
SymbolicEdge(rvsdg::input * output, rvsdg::output * input)
{
  auto color = "black";
  return GetDotName(output) + " -> " + GetDotName(input)
       + " [style=\"\", arrowhead=\"normal\", color=" + color
       + ", headlabel=<>, fontsize=10, labelangle=45, labeldistance=2.0, labelfontcolor=black, "
         "tooltip=\""
       + output->type().debug_string() + "\"];\n";
}

static bool
isForbiddenChar(char c)
{
  if (('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z') || ('0' <= c && c <= '9') || '_' == c)
  {
    return false;
  }
  return true;
}

std::string
StructuralNodeToDot(
    rvsdg::StructuralNode * structuralNode,
    std::unordered_map<rvsdg::output *, ViewColors> & outputColor,
    std::unordered_map<rvsdg::input *, ViewColors> & inputColor,
    std::unordered_map<rvsdg::output *, ViewColors> & tailLabel)
{

  std::ostringstream dot;
  dot << "subgraph cluster_sn" << hex((intptr_t)structuralNode) << " {\n";
  dot << "color=\"#ff8080\"\n";
  dot << "penwidth=6\n";
  dot << "label=\"" << structuralNode->GetOperation().debug_string() << "\"\n";
  dot << "labeljust=l\n";

  // input nodes
  for (size_t i = 0; i < structuralNode->ninputs(); ++i)
  {
    dot << StructuralInputToDot(
        structuralNode->input(i),
        GetDefaultColor<rvsdg::input>(inputColor, structuralNode->input(i)));
  }

  for (size_t i = 0; i < structuralNode->nsubregions(); ++i)
  {
    dot << RegionToDot(structuralNode->subregion(i), outputColor, inputColor, tailLabel);
  }

  for (size_t i = 0; i < structuralNode->ninputs(); ++i)
  {
    for (auto & argument : structuralNode->input(i)->arguments)
    {
      dot << SymbolicEdge(structuralNode->input(i), &argument);
    }
  }

  // output nodes
  for (size_t i = 0; i < structuralNode->noutputs(); ++i)
  {
    dot << StructuralOutputToDot(
        structuralNode->output(i),
        GetDefaultColor<rvsdg::output>(outputColor, structuralNode->output(i)));
    for (auto & result : structuralNode->output(i)->results)
    {
      dot << SymbolicEdge(&result, structuralNode->output(i));
    }
  }

  dot << "}\n";

  return dot.str();
}

std::string
SimpleNodeToDot(
    rvsdg::SimpleNode * simpleNode,
    std::unordered_map<rvsdg::output *, ViewColors> & outputColor,
    std::unordered_map<rvsdg::input *, ViewColors> & inputColor)
{
  auto SPACER = "                    <TD WIDTH=\"10\"></TD>\n";
  auto name = GetDotName(simpleNode);
  auto opname = simpleNode->GetOperation().debug_string();
  std::replace_if(opname.begin(), opname.end(), isForbiddenChar, '_');

  std::ostringstream inputs;
  // inputs
  for (size_t i = 0; i < simpleNode->ninputs(); ++i)
  {
    auto color = GetDefaultColor<rvsdg::input>(inputColor, simpleNode->input(i));
    if (i != 0)
    {
      inputs << SPACER;
    }
    inputs << "                    <TD PORT=\"i" << hex((intptr_t)simpleNode->input(i))
           << "\" BORDER=\"1\" CELLPADDING=\"1\" COLOR=\"" << color
           << "\"><FONT POINT-SIZE=\"10\" COLOR=\"" << color << "\"> i" << i << "</FONT></TD>\n";
  }

  std::ostringstream outputs;
  // outputs
  for (size_t i = 0; i < simpleNode->noutputs(); ++i)
  {
    auto color = GetDefaultColor<rvsdg::output>(outputColor, simpleNode->output(i));
    if (i != 0)
    {
      outputs << SPACER;
    }
    outputs << "                    <TD PORT=\"o" << hex((intptr_t)simpleNode->output(i))
            << "\" BORDER=\"1\" CELLPADDING=\"1\" COLOR=\"" << color
            << "\"><FONT POINT-SIZE=\"10\" COLOR=\"" << color << "\"> o" << i << "</FONT></TD>\n";
  }

  std::string color = "black";
  auto dot =
      name
      + " [shape=plaintext label=<\n"
        "<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\">\n"
        // inputs
        "    <TR>\n"
        "        <TD BORDER=\"0\">\n"
        "            <TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\">\n"
        "                <TR>\n"
        "                    <TD WIDTH=\"20\"></TD>\n"
      + inputs.str()
      + "                    <TD WIDTH=\"20\"></TD>\n"
        "                </TR>\n"
        "            </TABLE>\n"
        "        </TD>\n"
        "    </TR>\n"
        "    <TR>\n"
        "        <TD BORDER=\"3\" STYLE=\"ROUNDED\" CELLPADDING=\"4\">"
      + opname + "<BR/><FONT POINT-SIZE=\"10\">" + name
      + "</FONT></TD>\n"
        "    </TR>\n"
        "    <TR>\n"
        "        <TD BORDER=\"0\">\n"
        "            <TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\">\n"
        "                <TR>\n"
        "                    <TD WIDTH=\"20\"></TD>\n"
      + outputs.str()
      + "                    <TD WIDTH=\"20\"></TD>\n"
        "                </TR>\n"
        "            </TABLE>\n"
        "        </TD>\n"
        "    </TR>\n"
        "</TABLE>\n"
        "> fontcolor="
      + color + " color=" + color + "];\n";
  return dot;
}

std::string
RegionToDot(
    rvsdg::Region * region,
    std::unordered_map<rvsdg::output *, ViewColors> & outputColor,
    std::unordered_map<rvsdg::input *, ViewColors> & inputColor,
    std::unordered_map<rvsdg::output *, ViewColors> & tailLabel)
{
  std::ostringstream dot;
  dot << "subgraph cluster_reg" << hex((intptr_t)region) << " {\n";
  dot << "color=\"#80b3ff\"\n";
  dot << "penwidth=6\n";
  dot << "label=\"" << region->index() << " - " << hex((intptr_t)region) << "\"\n";

  // argument nodes
  dot << "{rank=source; ";
  for (size_t i = 0; i < region->narguments(); ++i)
  {
    dot << ArgumentToDot(
        region->argument(i),
        GetDefaultColor<rvsdg::output>(outputColor, region->argument(i)));
  }
  dot << "}\n";

  // nodes
  for (auto node : rvsdg::TopDownTraverser(region))
  {
    if (auto simpleNode = dynamic_cast<rvsdg::SimpleNode *>(node))
    {
      auto node_dot = SimpleNodeToDot(simpleNode, outputColor, inputColor);
      dot << node_dot;
    }
    else if (auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      auto node_dot = StructuralNodeToDot(structuralNode, outputColor, inputColor, tailLabel);
      dot << node_dot;
    }

    for (size_t i = 0; i < node->ninputs(); ++i)
    {
      dot << Edge(node->input(i)->origin(), node->input(i), tailLabel);
    }
  }

  // result nodes
  dot << "{rank=sink; ";
  for (size_t i = 0; i < region->nresults(); ++i)
  {
    dot << ResultToDot(
        region->result(i),
        GetDefaultColor<rvsdg::input>(inputColor, region->result(i)));
  }
  dot << "}\n";
  for (size_t i = 0; i < region->nresults(); ++i)
  {
    dot << Edge(region->result(i)->origin(), region->result(i), tailLabel);
    if (auto be = dynamic_cast<backedge_result *>(region->result(i)))
    {
      dot << Edge(be->argument(), be, tailLabel, true);
    }
    else if (
        region->result(i)->output()
        && rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*region->result(i)->output()))
    {
      auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*region->result(i)->output());
      auto loopvar = theta->MapOutputLoopVar(*region->result(i)->output());
      dot << Edge(loopvar.pre, loopvar.post, tailLabel, true);
    }
  }

  dot << "}\n";

  return dot.str();
}

std::string
ToDot(
    rvsdg::Region * region,
    std::unordered_map<rvsdg::output *, ViewColors> & outputColor,
    std::unordered_map<rvsdg::input *, ViewColors> & inputColor,
    std::unordered_map<rvsdg::output *, ViewColors> & tailLabel)
{
  std::ostringstream dot;
  dot << "digraph G {\n";
  dot << RegionToDot(region, outputColor, inputColor, tailLabel);
  dot << "}\n";
  return dot.str();
}

void
ViewDot(
    rvsdg::Region * region,
    FILE * out,
    std::unordered_map<rvsdg::output *, ViewColors> & outputColor,
    std::unordered_map<rvsdg::input *, ViewColors> & inputColor,
    std::unordered_map<rvsdg::output *, ViewColors> & tailLabel)
{
  fputs(ToDot(region, outputColor, inputColor, tailLabel).c_str(), out);
  fflush(out);
}

void
ViewDot(rvsdg::Region * region, FILE * out)
{
  std::unordered_map<rvsdg::output *, ViewColors> outputColor;
  std::unordered_map<rvsdg::input *, ViewColors> inputColor;
  std::unordered_map<rvsdg::output *, ViewColors> tailLabel;
  ViewDot(region, out, outputColor, inputColor, tailLabel);
}

void
DumpDot(llvm::RvsdgModule & rvsdgModule, const std::string & file_name)
{
  DumpDot(&rvsdgModule.Rvsdg().GetRootRegion(), file_name);
}

void
DumpDot(
    llvm::RvsdgModule & rvsdgModule,
    const std::string & file_name,
    std::unordered_map<rvsdg::output *, ViewColors> outputColor,
    std::unordered_map<rvsdg::input *, ViewColors> inputColor,
    std::unordered_map<rvsdg::output *, ViewColors> tailLabel)
{
  DumpDot(&rvsdgModule.Rvsdg().GetRootRegion(), file_name, outputColor, inputColor, tailLabel);
}

void
DumpDot(rvsdg::Region * region, const std::string & file_name)
{
  auto dot_file = fopen(file_name.c_str(), "w");
  ViewDot(region, dot_file);
  fclose(dot_file);
}

void
DumpDot(
    rvsdg::Region * region,
    const std::string & file_name,
    std::unordered_map<rvsdg::output *, ViewColors> outputColor,
    std::unordered_map<rvsdg::input *, ViewColors> inputColor,
    std::unordered_map<rvsdg::output *, ViewColors> tailLabel)
{
  auto dot_file = fopen(file_name.c_str(), "w");
  ViewDot(region, dot_file, outputColor, inputColor, tailLabel);
  fclose(dot_file);
}

void
DotToSvg(const std::string & file_name)
{
  auto cmd = "dot -Tsvg -O " + file_name;
  if (system(cmd.c_str()))
    exit(EXIT_FAILURE);
}

} // namespace jlm::hls

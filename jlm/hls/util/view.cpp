/*
 * Copyright 2010 2011 2012 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
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

static inline std::string
hex(size_t i)
{
  std::stringstream stream;
  stream << std::hex << i;
  return stream.str();
}

std::string
get_dot_name(rvsdg::Node * node)
{
  return util::strfmt("n", hex((intptr_t)node));
}

std::string
get_dot_name(rvsdg::output * output)
{
  if (dynamic_cast<rvsdg::RegionArgument *>(output))
  {
    return util::strfmt("a", hex((intptr_t)output), ":", "default");
  }

  if (auto no = dynamic_cast<rvsdg::SimpleOutput *>(output))
  {
    return util::strfmt(get_dot_name(no->node()), ":", "o", hex((intptr_t)output));
  }
  else if (dynamic_cast<rvsdg::StructuralOutput *>(output))
  {
    return util::strfmt("so", hex((intptr_t)output), ":", "default");
  }
  JLM_UNREACHABLE("not implemented");
}

template<class T>
std::string
get_default_color(std::unordered_map<T *, std::string> & map, T * elem, std::string def = "black")
{
  auto f = map.find(elem);
  if (f == map.end())
  {
    return def;
  }
  return f->second;
}

template<class T>
std::string
get_default_label(std::unordered_map<T *, std::string> & map, T * elem, std::string def = "")
{
  auto f = map.find(elem);
  if (f == map.end())
  {
    return def;
  }
  return f->second;
}

std::string
get_dot_name(rvsdg::input * input)
{
  if (dynamic_cast<rvsdg::RegionResult *>(input))
  {
    return util::strfmt("r", hex((intptr_t)input), ":", "default");
  }
  if (auto ni = dynamic_cast<rvsdg::SimpleInput *>(input))
  {
    return util::strfmt(get_dot_name(ni->node()), ":", "i", hex((intptr_t)input));
  }
  else if (dynamic_cast<rvsdg::StructuralInput *>(input))
  {
    return util::strfmt("si", hex((intptr_t)input), ":", "default");
  }
  JLM_UNREACHABLE("not implemented");
}

std::string
port_to_dot(const std::string & display_name, const std::string & dot_name, std::string & color)
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
      + dot_name + "\" color=" + color + " fontcolor=" + color + "];\n";
  return dot;
}

std::string
argument_to_dot(rvsdg::RegionArgument * argument, std::string color)
{
  auto display_name = util::strfmt("a", argument->index());
  auto dot_name = util::strfmt("a", hex((intptr_t)argument));
  return port_to_dot(display_name, dot_name, color);
}

std::string
result_to_dot(rvsdg::RegionResult * result, std::string color)
{
  auto display_name = util::strfmt("r", result->index());
  auto dot_name = util::strfmt("r", hex((intptr_t)result));
  return port_to_dot(display_name, dot_name, color);
}

std::string
structural_input_to_dot(rvsdg::StructuralInput * structuralInput, std::string color)
{
  auto display_name = util::strfmt("si", structuralInput->index());
  auto dot_name = util::strfmt("si", hex((intptr_t)structuralInput));
  return port_to_dot(display_name, dot_name, color);
}

std::string
structural_output_to_dot(rvsdg::StructuralOutput * structuralOutput, std::string color)
{
  auto display_name = util::strfmt("so", structuralOutput->index());
  auto dot_name = util::strfmt("so", hex((intptr_t)structuralOutput));
  return port_to_dot(display_name, dot_name, color);
}

std::string
edge(rvsdg::output * output, rvsdg::input * input, std::unordered_map<rvsdg::output *, std::string> & tail_label, bool back_edge = false)
{
  auto color = "black";
  auto tl = get_default_label(tail_label, output);
  if (!back_edge)
  {
    return get_dot_name(output) + " -> " + get_dot_name(input)
         + " [style=\"\", arrowhead=\"normal\", color=" + color
         + ", headlabel=<>, fontsize=15, labelangle=45, labeldistance=2.0, labelfontcolor=blue, "
           "tooltip=\""
         + output->type().debug_string() + "\", taillabel=\""+tl+"\"];\n";
  }
  return get_dot_name(input) + " -> " + get_dot_name(output)
       + " [style=\"\", arrowhead=\"normal\", color=" + color
       + ", headlabel=<>, fontsize=15, labelangle=45, labeldistance=2.0, labelfontcolor=blue, "
         "constraint=false, tooltip=\""
       + output->type().debug_string() + "\", taillabel=\""+tl+"\"];\n";
}

std::string
symbolic_edge(rvsdg::input * output, rvsdg::output * input)
{
  auto color = "black";
  return get_dot_name(output) + " -> " + get_dot_name(input)
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
structural_node_to_dot(
    rvsdg::StructuralNode * structuralNode,
    std::unordered_map<rvsdg::output *, std::string> & o_color,
    std::unordered_map<rvsdg::input *, std::string> & i_color,
    std::unordered_map<rvsdg::output *, std::string> & tail_label)
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
    dot << structural_input_to_dot(
        structuralNode->input(i),
        get_default_color<rvsdg::input>(i_color, structuralNode->input(i)));
  }

//  if (structuralNode->ninputs() > 1)
//  {
//
//    // order inputs horizontally
//    dot << "{rank=source; ";
//    for (size_t i = 0; i < structuralNode->ninputs(); ++i)
//    {
//      if (i > 0)
//      {
//        dot << " -> ";
//      }
//      dot << get_dot_name(structuralNode->input(i));
//    }
//    dot << "[style = invis]}\n";
//  }

  for (size_t i = 0; i < structuralNode->nsubregions(); ++i)
  {
    dot << region_to_dot(structuralNode->subregion(i), o_color, i_color, tail_label);
  }

  for (size_t i = 0; i < structuralNode->ninputs(); ++i)
  {
    for (auto & argument : structuralNode->input(i)->arguments)
    {
      dot << symbolic_edge(structuralNode->input(i), &argument);
    }
  }

  // output nodes
  for (size_t i = 0; i < structuralNode->noutputs(); ++i)
  {
    dot << structural_output_to_dot(
        structuralNode->output(i),
        get_default_color<rvsdg::output>(o_color, structuralNode->output(i)));
    for (auto & result : structuralNode->output(i)->results)
    {
      dot << symbolic_edge(&result, structuralNode->output(i));
    }
  }
//  if (structuralNode->noutputs() > 1)
//  {
//    // order outputs horizontally
//    dot << "{rank=sink; ";
//    for (size_t i = 0; i < structuralNode->noutputs(); ++i)
//    {
//      if (i > 0)
//      {
//        dot << " -> ";
//      }
//      dot << get_dot_name(structuralNode->output(i));
//    }
//    dot << "[style = invis]}\n";
//  }

  dot << "}\n";

  return dot.str();
}

std::string
simple_node_to_dot(
    rvsdg::SimpleNode * simpleNode,
    std::unordered_map<rvsdg::output *, std::string> & o_color,
    std::unordered_map<rvsdg::input *, std::string> & i_color)
{
  auto SPACER = "                    <TD WIDTH=\"10\"></TD>\n";
  auto name = get_dot_name(simpleNode);
  auto opname = simpleNode->GetOperation().debug_string();
  std::replace_if(opname.begin(), opname.end(), isForbiddenChar, '_');

  std::ostringstream inputs;
  // inputs
  for (size_t i = 0; i < simpleNode->ninputs(); ++i)
  {
    auto color = get_default_color<rvsdg::input>(i_color, simpleNode->input(i));
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
    auto color = get_default_color<rvsdg::output>(o_color, simpleNode->output(i));
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
region_to_dot(
    rvsdg::Region * region,
    std::unordered_map<rvsdg::output *, std::string> & o_color,
    std::unordered_map<rvsdg::input *, std::string> & i_color,
    std::unordered_map<rvsdg::output *, std::string> & tail_label)
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
    dot << argument_to_dot(
        region->argument(i),
        get_default_color<rvsdg::output>(o_color, region->argument(i)));
  }
  dot << "}\n";

//    if (region->narguments() > 1)
//    {
//      // order arguments horizontally
//      dot << "{rank=source; ";
//      for (size_t i = 0; i < region->narguments(); ++i)
//      {
//        if (i > 0)
//        {
//          dot << " -> ";
//        }
//        dot << get_dot_name(region->argument(i));
//      }
//      dot << "[style = invis]}\n";
//    }

  // nodes
  for (auto node : rvsdg::TopDownTraverser(region))
  {
    if (auto simpleNode = dynamic_cast<rvsdg::SimpleNode *>(node))
    {
      auto node_dot = simple_node_to_dot(simpleNode, o_color, i_color);
      dot << node_dot;
    }
    else if (auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      auto node_dot = structural_node_to_dot(structuralNode, o_color, i_color, tail_label);
      dot << node_dot;
    }

    for (size_t i = 0; i < node->ninputs(); ++i)
    {
      dot << edge(node->input(i)->origin(), node->input(i), tail_label);
    }
  }

  // result nodes
  dot << "{rank=sink; ";
  for (size_t i = 0; i < region->nresults(); ++i)
  {
    dot << result_to_dot(
        region->result(i),
        get_default_color<rvsdg::input>(i_color, region->result(i)));
  }
  dot << "}\n";
  for (size_t i = 0; i < region->nresults(); ++i)
  {
    dot << edge(region->result(i)->origin(), region->result(i), tail_label);
    if (auto be = dynamic_cast<backedge_result *>(region->result(i)))
    {
      dot << edge(be->argument(), be, tail_label, true);
    }
    else if (
        region->result(i)->output()
        && rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*region->result(i)->output()))
    {
      auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*region->result(i)->output());
      auto loopvar = theta->MapOutputLoopVar(*region->result(i)->output());
      dot << edge(loopvar.pre, loopvar.post, tail_label, true);
    }
  }

//    if (region->nresults() > 1)
//    {
//      // order results horizontally
//      dot << "{rank=sink; ";
//      for (size_t i = 0; i < region->nresults(); ++i)
//      {
//        if (i > 0)
//        {
//          dot << " -> ";
//        }
//        dot << get_dot_name(region->result(i));
//      }
//      dot << "[style = invis]}\n";
//    }

  dot << "}\n";

  return dot.str();
}

std::string
to_dot(
    rvsdg::Region * region,
    std::unordered_map<rvsdg::output *, std::string> & o_color,
    std::unordered_map<rvsdg::input *, std::string> & i_color,
    std::unordered_map<rvsdg::output *, std::string> & tail_label)
{
  std::ostringstream dot;
  dot << "digraph G {\n";
  dot << region_to_dot(region, o_color, i_color, tail_label);
  dot << "}\n";
  return dot.str();
}

void
view_dot(
    rvsdg::Region * region,
    FILE * out,
    std::unordered_map<rvsdg::output *, std::string> & o_color,
    std::unordered_map<rvsdg::input *, std::string> & i_color,
    std::unordered_map<rvsdg::output *, std::string> & tail_label)
{
  fputs(to_dot(region, o_color, i_color, tail_label).c_str(), out);
  fflush(out);
}

void
view_dot(rvsdg::Region * region, FILE * out)
{
  std::unordered_map<rvsdg::output *, std::string> o_color;
  std::unordered_map<rvsdg::input *, std::string> i_color;
  std::unordered_map<rvsdg::output *, std::string> tail_label;
  view_dot(region, out, o_color, i_color, tail_label);
}

void
dump_dot(llvm::RvsdgModule & rvsdgModule, const std::string & file_name)
{
  dump_dot(&rvsdgModule.Rvsdg().GetRootRegion(), file_name);
}

void
dump_dot(
    llvm::RvsdgModule & rvsdgModule,
    const std::string & file_name,
    std::unordered_map<rvsdg::output *, std::string> o_color,
    std::unordered_map<rvsdg::input *, std::string> i_color,
    std::unordered_map<rvsdg::output *, std::string> tail_label)
{
  dump_dot(&rvsdgModule.Rvsdg().GetRootRegion(), file_name, o_color, i_color, tail_label);
}

void
dump_dot(rvsdg::Region * region, const std::string & file_name)
{
  auto dot_file = fopen(file_name.c_str(), "w");
  view_dot(region, dot_file);
  fclose(dot_file);
}

void
dump_dot(
    rvsdg::Region * region,
    const std::string & file_name,
    std::unordered_map<rvsdg::output *, std::string> o_color,
    std::unordered_map<rvsdg::input *, std::string> i_color,
    std::unordered_map<rvsdg::output *, std::string> tail_label)
{
  auto dot_file = fopen(file_name.c_str(), "w");
  view_dot(region, dot_file, o_color, i_color, tail_label);
  fclose(dot_file);
}

void
dot_to_svg(const std::string & file_name)
{
  auto cmd = "dot -Tsvg -O " + file_name;
  if (system(cmd.c_str()))
    exit(EXIT_FAILURE);
}

} // namespace jlm::hls

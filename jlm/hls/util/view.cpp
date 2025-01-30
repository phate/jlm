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
  return jlm::util::strfmt("n", hex((intptr_t)node));
}

std::string
get_dot_name(jlm::rvsdg::output * output)
{
  if (dynamic_cast<rvsdg::RegionArgument *>(output))
  {
    return jlm::util::strfmt("a", hex((intptr_t)output), ":", "default");
  }
  else if (auto no = dynamic_cast<jlm::rvsdg::simple_output *>(output))
  {
    return jlm::util::strfmt(get_dot_name(no->node()), ":", "o", hex((intptr_t)output));
  }
  else if (dynamic_cast<rvsdg::StructuralOutput *>(output))
  {
    return jlm::util::strfmt("so", hex((intptr_t)output), ":", "default");
  }
  JLM_UNREACHABLE("not implemented");
}

std::string
get_dot_name(jlm::rvsdg::input * input)
{
  if (dynamic_cast<rvsdg::RegionResult *>(input))
  {
    return jlm::util::strfmt("r", hex((intptr_t)input), ":", "default");
  }
  else if (auto ni = dynamic_cast<jlm::rvsdg::simple_input *>(input))
  {
    return jlm::util::strfmt(get_dot_name(ni->node()), ":", "i", hex((intptr_t)input));
  }
  else if (dynamic_cast<rvsdg::StructuralInput *>(input))
  {
    return jlm::util::strfmt("si", hex((intptr_t)input), ":", "default");
  }
  JLM_UNREACHABLE("not implemented");
}

std::string
port_to_dot(const std::string & display_name, const std::string & dot_name)
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
      + dot_name + "\"];\n";
  return dot;
}

std::string
argument_to_dot(rvsdg::RegionArgument * argument)
{
  auto display_name = jlm::util::strfmt("a", argument->index());
  auto dot_name = jlm::util::strfmt("a", hex((intptr_t)argument));
  return port_to_dot(display_name, dot_name);
}

std::string
result_to_dot(rvsdg::RegionResult * result)
{
  auto display_name = jlm::util::strfmt("r", result->index());
  auto dot_name = jlm::util::strfmt("r", hex((intptr_t)result));
  return port_to_dot(display_name, dot_name);
}

std::string
structural_input_to_dot(rvsdg::StructuralInput * structuralInput)
{
  auto display_name = jlm::util::strfmt("si", structuralInput->index());
  auto dot_name = jlm::util::strfmt("si", hex((intptr_t)structuralInput));
  return port_to_dot(display_name, dot_name);
}

std::string
structural_output_to_dot(rvsdg::StructuralOutput * structuralOutput)
{
  auto display_name = jlm::util::strfmt("so", structuralOutput->index());
  auto dot_name = jlm::util::strfmt("so", hex((intptr_t)structuralOutput));
  return port_to_dot(display_name, dot_name);
}

std::string
edge(jlm::rvsdg::output * output, jlm::rvsdg::input * input, bool back_edge = false)
{
  auto color = "black";
  if (!back_edge)
  {
    return get_dot_name(output) + " -> " + get_dot_name(input)
         + " [style=\"\", arrowhead=\"normal\", color=" + color
         + ", headlabel=<>, fontsize=10, labelangle=45, labeldistance=2.0, labelfontcolor=black, "
           "tooltip=\""
         + output->type().debug_string() + "\"];\n";
  }
  return get_dot_name(input) + " -> " + get_dot_name(output)
       + " [style=\"\", arrowhead=\"normal\", color=" + color
       + ", headlabel=<>, fontsize=10, labelangle=45, labeldistance=2.0, labelfontcolor=black, "
         "constraint=false, tooltip=\""
       + output->type().debug_string() + "\"];\n";
}

std::string
symbolic_edge(jlm::rvsdg::input * output, jlm::rvsdg::output * input)
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
structural_node_to_dot(rvsdg::StructuralNode * structuralNode)
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
    dot << structural_input_to_dot(structuralNode->input(i));
  }

  if (structuralNode->ninputs() > 1)
  {

    // order inputs horizontally
    dot << "{rank=source; ";
    for (size_t i = 0; i < structuralNode->ninputs(); ++i)
    {
      if (i > 0)
      {
        dot << " -> ";
      }
      dot << get_dot_name(structuralNode->input(i));
    }
    dot << "[style = invis]}\n";
  }

  for (size_t i = 0; i < structuralNode->nsubregions(); ++i)
  {
    dot << jlm::hls::region_to_dot(structuralNode->subregion(i));
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
    dot << structural_output_to_dot(structuralNode->output(i));
    for (auto & result : structuralNode->output(i)->results)
    {
      dot << symbolic_edge(&result, structuralNode->output(i));
    }
  }
  if (structuralNode->noutputs() > 1)
  {
    // order results horizontally
    dot << "{rank=sink; ";
    for (size_t i = 0; i < structuralNode->noutputs(); ++i)
    {
      if (i > 0)
      {
        dot << " -> ";
      }
      dot << get_dot_name(structuralNode->output(i));
    }
    dot << "[style = invis]}\n";
  }

  dot << "}\n";

  return dot.str();
}

std::string
simple_node_to_dot(jlm::rvsdg::SimpleNode * simpleNode)
{
  auto SPACER = "                    <TD WIDTH=\"10\"></TD>\n";
  auto name = get_dot_name(simpleNode);
  auto opname = simpleNode->GetOperation().debug_string();
  std::replace_if(opname.begin(), opname.end(), isForbiddenChar, '_');

  std::ostringstream inputs;
  // inputs
  for (size_t i = 0; i < simpleNode->ninputs(); ++i)
  {
    if (i != 0)
    {
      inputs << SPACER;
    }
    inputs << "                    <TD PORT=\"i" << hex((intptr_t)simpleNode->input(i))
           << "\" BORDER=\"1\" CELLPADDING=\"1\"><FONT POINT-SIZE=\"10\"> i" << i
           << "</FONT></TD>\n";
  }

  std::ostringstream outputs;
  // inputs
  for (size_t i = 0; i < simpleNode->noutputs(); ++i)
  {
    if (i != 0)
    {
      outputs << SPACER;
    }
    outputs << "                    <TD PORT=\"o" << hex((intptr_t)simpleNode->output(i))
            << "\" BORDER=\"1\" CELLPADDING=\"1\"><FONT POINT-SIZE=\"10\"> o" << i
            << "</FONT></TD>\n";
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
region_to_dot(rvsdg::Region * region)
{
  std::ostringstream dot;
  dot << "subgraph cluster_reg" << hex((intptr_t)region) << " {\n";
  dot << "color=\"#80b3ff\"\n";
  dot << "penwidth=6\n";
  dot << "label=\"" << region->index() << "\"\n";

  // argument nodes
  for (size_t i = 0; i < region->narguments(); ++i)
  {
    dot << argument_to_dot(region->argument(i));
  }

  if (region->narguments() > 1)
  {
    // order arguments horizontally
    dot << "{rank=source; ";
    for (size_t i = 0; i < region->narguments(); ++i)
    {
      if (i > 0)
      {
        dot << " -> ";
      }
      dot << get_dot_name(region->argument(i));
    }
    dot << "[style = invis]}\n";
  }

  // nodes
  for (auto node : rvsdg::TopDownTraverser(region))
  {
    if (auto simpleNode = dynamic_cast<jlm::rvsdg::SimpleNode *>(node))
    {
      auto node_dot = simple_node_to_dot(simpleNode);
      dot << node_dot;
    }
    else if (auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      auto node_dot = structural_node_to_dot(structuralNode);
      dot << node_dot;
    }

    for (size_t i = 0; i < node->ninputs(); ++i)
    {
      dot << edge(node->input(i)->origin(), node->input(i));
    }
  }

  // result nodes
  for (size_t i = 0; i < region->nresults(); ++i)
  {
    dot << result_to_dot(region->result(i));
    dot << edge(region->result(i)->origin(), region->result(i));
    if (auto be = dynamic_cast<jlm::hls::backedge_result *>(region->result(i)))
    {
      dot << edge(be->argument(), be, true);
    }
    else if (auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*region->result(i)))
    {
      auto loopvar = theta->MapOutputLoopVar(*region->result(i)->output());
      dot << edge(loopvar.pre, loopvar.post, true);
    }
  }

  if (region->nresults() > 1)
  {
    // order results horizontally
    dot << "{rank=sink; ";
    for (size_t i = 0; i < region->nresults(); ++i)
    {
      if (i > 0)
      {
        dot << " -> ";
      }
      dot << get_dot_name(region->result(i));
    }
    dot << "[style = invis]}\n";
  }

  dot << "}\n";

  return dot.str();
}

std::string
to_dot(rvsdg::Region * region)
{
  std::ostringstream dot;
  dot << "digraph G {\n";
  dot << jlm::hls::region_to_dot(region);
  dot << "}\n";
  return dot.str();
}

void
view_dot(rvsdg::Region * region, FILE * out)
{
  fputs(jlm::hls::to_dot(region).c_str(), out);
  fflush(out);
}

void
dump_dot(jlm::llvm::RvsdgModule & rvsdgModule, const std::string & file_name)
{
  auto dot_file = fopen(file_name.c_str(), "w");
  jlm::hls::view_dot(&rvsdgModule.Rvsdg().GetRootRegion(), dot_file);
  fclose(dot_file);
}

} // namespace jlm::hls

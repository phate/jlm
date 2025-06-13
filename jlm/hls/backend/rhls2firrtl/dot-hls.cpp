/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rhls2firrtl/dot-hls.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include <algorithm>
#include <set>

namespace jlm::hls
{

std::string
DotHLS::extension()
{
  return ".dot";
}

std::string
DotHLS::GetText(llvm::RvsdgModule & rm)
{
  return subregion_to_dot(get_hls_lambda(rm)->subregion());
}

std::string
DotHLS::argument_to_dot(rvsdg::RegionArgument * port)
{
  auto name = get_port_name(port);

  auto dot =
      name
      + " [shape=plaintext label=<\n"
        "            <TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\">\n"
        "                <TR>\n"
        "                    <TD BORDER=\"1\" CELLPADDING=\"1\"><FONT POINT-SIZE=\"10\">"
      + name
      + "</FONT></TD>\n"
        "                </TR>\n"
        "            </TABLE>\n"
        ">];\n";
  return dot;
}

std::string
DotHLS::result_to_dot(rvsdg::RegionResult * port)
{
  auto name = get_port_name(port);

  auto dot =
      "{rank=sink; " + name
      + " [shape=plaintext label=<\n"
        "            <TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\">\n"
        "                <TR>\n"
        "                    <TD BORDER=\"1\" CELLPADDING=\"1\"><FONT POINT-SIZE=\"10\">"
      + name
      + "</FONT></TD>\n"
        "                </TR>\n"
        "            </TABLE>\n"
        ">];}\n";
  return dot;
}

std::string
DotHLS::node_to_dot(const rvsdg::Node * node)
{
  auto SPACER = "                    <TD WIDTH=\"10\"></TD>\n";
  auto name = get_node_name(node);
  auto opname = node->DebugString();
  std::replace_if(opname.begin(), opname.end(), isForbiddenChar, '_');

  std::string inputs;
  for (size_t i = 0; i < node->ninputs(); ++i)
  {
    if (i != 0)
    {
      inputs += SPACER;
    }
    auto in = get_port_name(node->input(i));
    inputs += "                    <TD PORT=\"" + in
            + "\" BORDER=\"1\" CELLPADDING=\"1\"><FONT POINT-SIZE=\"10\">" + in + "</FONT></TD>\n";
  }

  std::string outputs;
  for (size_t i = 0; i < node->noutputs(); ++i)
  {
    if (i != 0)
    {
      outputs += SPACER;
    }
    auto out = get_port_name(node->output(i));
    outputs += "                    <TD PORT=\"" + out
             + "\" BORDER=\"1\" CELLPADDING=\"1\"><FONT POINT-SIZE=\"10\">" + out
             + "</FONT></TD>\n";
  }

  std::string color = "black";
  if (jlm::rvsdg::is<hls::buffer_op>(node))
  {
    color = "blue";
  }
  else if (jlm::rvsdg::is<ForkOperation>(node))
  {
    color = "grey";
  }
  else if (jlm::rvsdg::is<hls::sink_op>(node))
  {
    color = "grey";
  }
  else if (jlm::rvsdg::is<hls::branch_op>(node))
  {
    color = "green";
  }
  else if (jlm::rvsdg::is<hls::mux_op>(node))
  {
    color = "darkred";
  }
  else if (jlm::rvsdg::is<hls::merge_op>(node))
  {
    color = "pink";
  }
  else if (jlm::rvsdg::is<hls::trigger_op>(node) || hls::is_constant(node))
  {
    color = "orange";
  }

  // dot inspired by
  // https://stackoverflow.com/questions/42157650/moving-graphviz-edge-out-of-the-way
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
      + inputs
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
      + outputs
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
DotHLS::edge(std::string src, std::string snk, const jlm::rvsdg::Type & type, bool back)
{
  auto color = "black";
  JLM_ASSERT(src != "" && snk != "");
  if (dynamic_cast<const rvsdg::ControlType *>(&type))
  {
    color = "green";
  }
  else if (dynamic_cast<const llvm::MemoryStateType *>(&type))
  {
    color = "blue";
  }
  if (!back)
  {
    return src + " -> " + snk + " [style=\"\", arrowhead=\"normal\", color=" + color
         + ", headlabel=<>, fontsize=10, labelangle=45, labeldistance=2.0, "
           "labelfontcolor=black, penwidth=3];\n";
  }
  // implement back edges by setting constraint to false
  return src + " -> " + snk + " [style=\"\", arrowhead=\"normal\", color=" + color
       + ", headlabel=<>, fontsize=10, labelangle=45, labeldistance=2.0, labelfontcolor=black, "
         "penwidth=3, constraint=false];\n";
  //	return snk + " -> " + src + " [style=\"\", arrowhead=\"normal\", color=" + color +
  //		   ", headlabel=<>, fontsize=10, labelangle=45, labeldistance=2.0, labelfontcolor=black,
  //  dir=back];\n";
}

std::string
DotHLS::loop_to_dot(hls::loop_node * ln)
{
  auto sr = ln->subregion();
  std::ostringstream dot;
  dot << "subgraph cluster_loop_" << loop_ctr++ << " {\n";
  dot << "color=\"#ff8080\"\n";

  std::set<jlm::rvsdg::Output *> back_outputs;
  std::set<rvsdg::Node *> top_nodes; // no artificial top nodes for now
  for (size_t i = 0; i < sr->narguments(); ++i)
  {
    auto arg = sr->argument(i);
    // arguments without an input produce a cycle
    if (arg->input() == nullptr)
    {
      back_outputs.insert(arg);
    }
  }

  for (auto node : rvsdg::TopDownTraverser(sr))
  {
    if (dynamic_cast<jlm::rvsdg::SimpleNode *>(node))
    {
      auto node_dot = node_to_dot(node);
      if (top_nodes.count(node))
      {
        dot << "{rank=source; \n";
        dot << node_dot;
        dot << "}\n";
      }
      else
      {
        dot << node_dot;
      }
    }
    else if (auto ln = dynamic_cast<hls::loop_node *>(node))
    {
      // need to prepare output here again, because inputs might not have been resolved yet, because
      // nodes in outer loop were not yet processed.
      prepare_loop_out_port(ln);
      dot << loop_to_dot(ln);
    }
    else
    {
      throw jlm::util::error(
          "Unimplemented op (unexpected structural node) : " + node->DebugString());
    }
  }

  // all loop muxes at one level
  dot << "{rank=same ";
  for (auto node : rvsdg::TopDownTraverser(sr))
  {
    auto mx = dynamic_cast<const hls::mux_op *>(&node->GetOperation());
    auto lc = dynamic_cast<const hls::loop_constant_buffer_op *>(&node->GetOperation());
    if ((mx && !mx->discarding && mx->loop) || lc)
    {
      dot << get_node_name(node) << " ";
    }
  }
  dot << "}\n";
  // all loop branches at one level
  dot << "{rank=same ";
  for (auto node : rvsdg::TopDownTraverser(sr))
  {
    auto br = dynamic_cast<const hls::branch_op *>(&node->GetOperation());
    if (br && br->loop)
    {
      dot << get_node_name(node) << " ";
    }
  }
  dot << "}\n";

  dot << "}\n";
  // do edges outside in order not to pull other nodes into the cluster
  for (auto node : rvsdg::TopDownTraverser(sr))
  {
    if (dynamic_cast<jlm::rvsdg::SimpleNode *>(node))
    {
      auto mx = dynamic_cast<const hls::mux_op *>(&node->GetOperation());
      auto node_name = get_node_name(node);
      for (size_t i = 0; i < node->ninputs(); ++i)
      {
        auto in_name = node_name + ":" + get_port_name(node->input(i));
        JLM_ASSERT(output_map.count(node->input(i)->origin()));
        auto origin = output_map[node->input(i)->origin()];
        // implement edge as back edge when it produces a cycle
        bool back = mx && !mx->discarding && mx->loop
                 && (/*i==0||*/ i == 2); // back_outputs.count(node->input(i)->origin());
        auto origin_out_node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*node->input(i)->origin());
        if (origin_out_node
            && dynamic_cast<const predicate_buffer_op *>(&origin_out_node->GetOperation()))
        {
          //
          back = true;
        }
        dot << edge(origin, in_name, *node->input(i)->Type(), back);
      }
    }
  }
  return dot.str();
}

void
DotHLS::prepare_loop_out_port(hls::loop_node * ln)
{
  // make sure all outputs are translated and available (necessary for argument/result cycles)

  auto sr = ln->subregion();
  // just translate outputs
  for (auto node : rvsdg::TopDownTraverser(sr))
  {
    if (dynamic_cast<jlm::rvsdg::SimpleNode *>(node))
    {
      auto node_name = get_node_name(node);
      for (size_t i = 0; i < node->noutputs(); ++i)
      {
        output_map[node->output(i)] = node_name + ":" + get_port_name(node->output(i));
      }
    }
    else if (auto oln = dynamic_cast<hls::loop_node *>(node))
    {
      prepare_loop_out_port(oln);
    }
    else
    {
      throw jlm::util::error(
          "Unimplemented op (unexpected structural node) : " + node->DebugString());
    }
  }
  for (size_t i = 0; i < sr->narguments(); ++i)
  {
    auto arg = sr->argument(i);
    auto ba = dynamic_cast<backedge_argument *>(arg);
    if (!ba)
    {
      JLM_ASSERT(arg->input() != nullptr);
      // map to input of loop
      output_map[arg] = output_map[arg->input()->origin()];
    }
    else
    {
      auto result = ba->result();
      JLM_ASSERT(*result->Type() == *arg->Type());
      // map to end of loop (origin of associated result)
      output_map[arg] = output_map[result->origin()];
    }
  }
  for (size_t i = 0; i < ln->noutputs(); ++i)
  {
    auto out = ln->output(i);
    JLM_ASSERT(out->results.size() == 1);
    output_map[out] = output_map[out->results.begin()->origin()];
  }
}

std::string
DotHLS::subregion_to_dot(rvsdg::Region * sr)
{
  std::ostringstream dot;
  dot << "digraph G {\n";
  for (size_t i = 0; i < sr->narguments(); ++i)
  {
    dot << argument_to_dot(sr->argument(i));
  }
  // order arguments horizontally
  dot << "{rank=source; ";
  for (size_t i = 0; i < sr->narguments(); ++i)
  {
    if (i > 0)
    {
      dot << " -> ";
    }
    dot << get_port_name(sr->argument(i));
  }
  dot << "[style = invis]}";

  for (size_t i = 0; i < sr->nresults(); ++i)
  {
    dot << result_to_dot(sr->result(i));
  }
  dot << "subgraph cluster_sub {\n";
  dot << "color=\"#80b3ff\"\n";
  dot << "penwidth=6\n";

  // process arguments
  for (size_t i = 0; i < sr->narguments(); ++i)
  {
    output_map[sr->argument(i)] = get_port_name(sr->argument(i));
  }
  // process nodes
  for (auto node : rvsdg::TopDownTraverser(sr))
  {
    if (dynamic_cast<jlm::rvsdg::SimpleNode *>(node))
    {
      auto node_dot = node_to_dot(node);
      dot << node_dot;
      auto node_name = get_node_name(node);
      for (size_t i = 0; i < node->ninputs(); ++i)
      {
        auto in_name = node_name + ":" + get_port_name(node->input(i));
        JLM_ASSERT(output_map.count(node->input(i)->origin()));
        auto origin = output_map[node->input(i)->origin()];
        dot << edge(origin, in_name, *node->input(i)->Type());
      }
      for (size_t i = 0; i < node->noutputs(); ++i)
      {
        output_map[node->output(i)] = node_name + ":" + get_port_name(node->output(i));
      }
    }
    else if (auto ln = dynamic_cast<hls::loop_node *>(node))
    {
      // the only structural nodes left are loop nodes
      prepare_loop_out_port(ln);
      dot << loop_to_dot(ln);
    }
    else
    {
      throw jlm::util::error(
          "Unimplemented op (unexpected structural node) : " + node->DebugString());
    }
  }
  // process results
  for (size_t i = 0; i < sr->nresults(); ++i)
  {
    auto origin = output_map[sr->result(i)->origin()];
    auto result = get_port_name(sr->result(i));
    dot << edge(origin, result, *sr->result(i)->Type());
  }
  dot << "}\n";
  dot << "}\n";
  return dot.str();
}

}

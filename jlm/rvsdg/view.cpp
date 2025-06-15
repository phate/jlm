/*
 * Copyright 2010 2011 2012 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

namespace jlm::rvsdg
{

static std::string
region_to_string(
    const rvsdg::Region * region,
    size_t depth,
    std::unordered_map<const Output *, std::string> &);

static std::string
indent(size_t depth)
{
  return std::string(depth * 2, ' ');
}

static std::string
create_port_name(
    const jlm::rvsdg::Output * port,
    std::unordered_map<const Output *, std::string> & map)
{
  std::string name = dynamic_cast<const rvsdg::RegionArgument *>(port) ? "a" : "o";
  name += jlm::util::strfmt(map.size());
  return name;
}

static std::string
node_to_string(
    const Node * node,
    size_t depth,
    std::unordered_map<const Output *, std::string> & map)
{
  std::string s(indent(depth));
  for (size_t n = 0; n < node->noutputs(); n++)
  {
    auto name = create_port_name(node->output(n), map);
    map[node->output(n)] = name;
    s = s + name + " ";
  }

  s += ":= " + node->DebugString() + " ";

  for (size_t n = 0; n < node->ninputs(); n++)
  {
    s += map[node->input(n)->origin()];
    if (n <= node->ninputs() - 1)
      s += " ";
  }
  s += "\n";

  if (auto snode = dynamic_cast<const rvsdg::StructuralNode *>(node))
  {
    for (size_t n = 0; n < snode->nsubregions(); n++)
      s += region_to_string(snode->subregion(n), depth + 1, map);
  }

  return s;
}

static std::string
region_header(const rvsdg::Region * region, std::unordered_map<const Output *, std::string> & map)
{
  std::string header("[");
  for (size_t n = 0; n < region->narguments(); n++)
  {
    auto argument = region->argument(n);
    auto pname = create_port_name(argument, map);
    map[argument] = pname;

    header += pname;
    if (argument->input())
      header += jlm::util::strfmt(" <= ", map[argument->input()->origin()]);

    if (n < region->narguments() - 1)
      header += ", ";
  }
  header += "]{";

  return header;
}

static std::string
region_body(
    const rvsdg::Region * region,
    size_t depth,
    std::unordered_map<const Output *, std::string> & map)
{
  std::vector<std::vector<const Node *>> context;
  for (const auto & node : region->Nodes())
  {
    if (node.depth() >= context.size())
      context.resize(node.depth() + 1);
    context[node.depth()].push_back(&node);
  }

  std::string body;
  for (const auto & nodes : context)
  {
    for (const auto & node : nodes)
      body += node_to_string(node, depth, map);
  }

  return body;
}

static std::string
region_footer(const rvsdg::Region * region, std::unordered_map<const Output *, std::string> & map)
{
  std::string footer("}[");
  for (size_t n = 0; n < region->nresults(); n++)
  {
    auto result = region->result(n);
    auto pname = map[result->origin()];

    if (result->output())
      footer += map[result->output()] + " <= ";
    footer += pname;

    if (n < region->nresults() - 1)
      footer += ", ";
  }
  footer += "]";

  return footer;
}

static std::string
region_to_string(
    const rvsdg::Region * region,
    size_t depth,
    std::unordered_map<const Output *, std::string> & map)
{
  std::string s;
  s = indent(depth) + region_header(region, map) + "\n";
  s = s + region_body(region, depth + 1, map);
  s = s + indent(depth) + region_footer(region, map) + "\n";
  return s;
}

std::string
view(const rvsdg::Region * region)
{
  std::unordered_map<const Output *, std::string> map;
  return view(region, map);
}

std::string
view(const rvsdg::Region * region, std::unordered_map<const Output *, std::string> & map)
{
  return region_to_string(region, 0, map);
}

void
view(const rvsdg::Region * region, FILE * out)
{
  fputs(view(region).c_str(), out);
  fflush(out);
}

/* xml */

static inline std::string
xml_header()
{
  return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
         "<rvsdg>\n";
}

static inline std::string
xml_footer()
{
  return "</rvsdg>\n";
}

static inline std::string
id(const jlm::rvsdg::Output * port)
{
  return jlm::util::strfmt("o", (intptr_t)port);
}

static inline std::string
id(const jlm::rvsdg::Input * port)
{
  return jlm::util::strfmt("i", (intptr_t)port);
}

static inline std::string
id(const Node * node)
{
  return jlm::util::strfmt("n", (intptr_t)node);
}

static inline std::string
id(const rvsdg::Region * region)
{
  return jlm::util::strfmt("r", (intptr_t)region);
}

static inline std::string
argument_tag(const std::string & id)
{
  return "<argument id=\"" + id + "\"/>\n";
}

static inline std::string
result_tag(const std::string & id)
{
  return "<result id=\"" + id + "\"/>\n";
}

static inline std::string
input_tag(const std::string & id)
{
  return "<input id=\"" + id + "\"/>\n";
}

static inline std::string
output_tag(const std::string & id)
{
  return "<output id=\"" + id + "\"/>\n";
}

static inline std::string
node_starttag(const std::string & id, const std::string & name, const std::string & type)
{
  return "<node id=\"" + id + "\" name=\"" + name + "\" type=\"" + type + "\">\n";
}

static inline std::string
node_endtag()
{
  return "</node>\n";
}

static inline std::string
region_starttag(const std::string & id)
{
  return "<region id=\"" + id + "\">\n";
}

static inline std::string
region_endtag(const std::string &)
{
  return "</region>\n";
}

static inline std::string
edge_tag(const std::string & srcid, const std::string & dstid)
{
  return "<edge source=\"" + srcid + "\" target=\"" + dstid + "\"/>\n";
}

static inline std::string
type(const Node * n)
{
  if (dynamic_cast<const GammaNode *>(n))
    return "gamma";

  if (dynamic_cast<const ThetaNode *>(n))
    return "theta";

  return "";
}

static std::string
convert_region(const jlm::rvsdg::Region * region);

static inline std::string
convert_simple_node(const jlm::rvsdg::SimpleNode * node)
{
  std::string s;

  s += node_starttag(id(node), node->DebugString(), "");
  for (size_t n = 0; n < node->ninputs(); n++)
    s += input_tag(id(node->input(n)));
  for (size_t n = 0; n < node->noutputs(); n++)
    s += output_tag(id(node->output(n)));
  s += node_endtag();

  for (size_t n = 0; n < node->noutputs(); n++)
  {
    auto output = node->output(n);
    for (const auto & user : *output)
      s += edge_tag(id(output), id(user));
  }

  return s;
}

static inline std::string
convert_structural_node(const rvsdg::StructuralNode * node)
{
  std::string s;
  s += node_starttag(id(node), "", type(node));

  for (size_t n = 0; n < node->ninputs(); n++)
    s += input_tag(id(node->input(n)));
  for (size_t n = 0; n < node->noutputs(); n++)
    s += output_tag(id(node->output(n)));

  for (size_t n = 0; n < node->nsubregions(); n++)
    s += convert_region(node->subregion(n));
  s += node_endtag();

  for (size_t n = 0; n < node->noutputs(); n++)
  {
    auto output = node->output(n);
    for (const auto & user : *output)
      s += edge_tag(id(output), id(user));
  }

  return s;
}

static inline std::string
convert_node(const Node * node)
{
  if (auto n = dynamic_cast<const SimpleNode *>(node))
    return convert_simple_node(n);

  if (auto n = dynamic_cast<const StructuralNode *>(node))
    return convert_structural_node(n);

  JLM_ASSERT(0);
  return "";
}

static inline std::string
convert_region(const rvsdg::Region * region)
{
  std::string s;
  s += region_starttag(id(region));

  for (size_t n = 0; n < region->narguments(); n++)
    s += argument_tag(id(region->argument(n)));

  for (const auto & node : region->Nodes())
    s += convert_node(&node);

  for (size_t n = 0; n < region->nresults(); n++)
    s += result_tag(id(region->result(n)));

  for (size_t n = 0; n < region->narguments(); n++)
  {
    auto argument = region->argument(n);
    for (const auto & user : *argument)
      s += edge_tag(id(argument), id(user));
  }

  s += region_endtag(id(region));

  return s;
}

std::string
to_xml(const rvsdg::Region * region)
{
  std::string s;
  s += xml_header();

  s += convert_region(region);

  s += xml_footer();
  return s;
}

void
view_xml(const rvsdg::Region * region, FILE * out)
{
  fputs(to_xml(region).c_str(), out);
  fflush(out);
}

}

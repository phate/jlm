/*
 * Copyright 2010 2011 2012 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
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
    const Region * region,
    const size_t depth,
    std::unordered_map<const Output *, std::string> & map)
{
  std::string body;
  for (const auto node : TopDownConstTraverser(region))
  {
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

}

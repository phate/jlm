/*
 * Copyright 2010 2011 2012 2013 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/tracker.hpp>

#include <algorithm>
#include <cxxabi.h>

namespace jlm::rvsdg
{

GraphImport::GraphImport(Graph & graph, std::shared_ptr<const rvsdg::Type> type, std::string name)
    : RegionArgument(&graph.GetRootRegion(), nullptr, std::move(type)),
      Name_(std::move(name))
{}

std::string
GraphImport::debug_string() const
{
  return util::strfmt("import[", Name_, "]");
}

GraphExport::GraphExport(rvsdg::output & origin, std::string name)
    : RegionResult(&origin.region()->graph()->GetRootRegion(), &origin, nullptr, origin.Type()),
      Name_(std::move(name))
{}

std::string
GraphExport::debug_string() const
{
  return util::strfmt("export[", Name_, "]");
}

Graph::~Graph()
{
  JLM_ASSERT(!has_active_trackers(this));
}

Graph::Graph()
    : Normalized_(false),
      RootRegion_(new Region(nullptr, this))
{}

std::unique_ptr<Graph>
Graph::Copy() const
{
  SubstitutionMap substitutionMap;
  auto graph = std::make_unique<Graph>();
  GetRootRegion().copy(&graph->GetRootRegion(), substitutionMap, true, true);
  return graph;
}

node_normal_form *
Graph::GetNodeNormalForm(const std::type_info & type) noexcept
{
  auto i = NodeNormalForms_.find(std::type_index(type));
  if (i != NodeNormalForms_.end())
    return i.ptr();

  const auto cinfo = dynamic_cast<const abi::__si_class_type_info *>(&type);
  auto parent_normal_form = cinfo ? GetNodeNormalForm(*cinfo->__base_type) : nullptr;

  std::unique_ptr<node_normal_form> nf(node_normal_form::create(type, parent_normal_form, this));

  node_normal_form * result = nf.get();
  NodeNormalForms_.insert(std::move(nf));

  return result;
}

std::vector<Node *>
Graph::ExtractTailNodes(const Graph & rvsdg)
{
  auto IsOnlyExported = [](const rvsdg::output & output)
  {
    auto IsRootRegionExport = [](const rvsdg::input * input)
    {
      if (!input->region()->IsRootRegion())
      {
        return false;
      }

      if (input::GetNode(*input))
      {
        return false;
      }

      return true;
    };

    return std::all_of(output.begin(), output.end(), IsRootRegionExport);
  };

  auto & rootRegion = rvsdg.GetRootRegion();

  std::vector<Node *> nodes;
  for (auto & bottomNode : rootRegion.BottomNodes())
  {
    nodes.push_back(&bottomNode);
  }

  for (size_t n = 0; n < rootRegion.nresults(); n++)
  {
    auto output = rootRegion.result(n)->origin();
    if (IsOnlyExported(*output))
    {
      nodes.push_back(output::GetNode(*output));
    }
  }

  return nodes;
}

}

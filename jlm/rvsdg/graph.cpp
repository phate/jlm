/*
 * Copyright 2010 2011 2012 2013 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/tracker.hpp>

#include <algorithm>

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

GraphImport &
GraphImport::Copy(Region & region, StructuralInput *)
{
  return Create(*region.graph(), Type(), Name());
}

GraphImport &
GraphImport::Create(Graph & graph, std::shared_ptr<const rvsdg::Type> type, std::string name)
{
  auto graphImport = new GraphImport(graph, std::move(type), std::move(name));
  graph.GetRootRegion().append_argument(graphImport);
  return *graphImport;
}

GraphExport::GraphExport(rvsdg::Output & origin, std::string name)
    : RegionResult(&origin.region()->graph()->GetRootRegion(), &origin, nullptr, origin.Type()),
      Name_(std::move(name))
{}

std::string
GraphExport::debug_string() const
{
  return util::strfmt("export[", Name_, "]");
}

GraphExport &
GraphExport::Copy(Output & origin, StructuralOutput * output)
{
  JLM_ASSERT(output == nullptr);
  return Create(origin, Name());
}

GraphExport &
GraphExport::Create(Output & origin, std::string name)
{
  auto graphExport = new GraphExport(origin, std::move(name));
  origin.region()->graph()->GetRootRegion().append_result(graphExport);
  return *graphExport;
}

Graph::~Graph()
{
  JLM_ASSERT(!has_active_trackers(this));
}

Graph::Graph()
    : RootRegion_(new Region(nullptr, this))
{}

std::unique_ptr<Graph>
Graph::Copy() const
{
  SubstitutionMap substitutionMap;
  auto graph = std::make_unique<Graph>();
  GetRootRegion().copy(&graph->GetRootRegion(), substitutionMap, true, true);
  return graph;
}

std::vector<Node *>
Graph::ExtractTailNodes(const Graph & rvsdg)
{
  auto IsOnlyExported = [](const rvsdg::Output & output)
  {
    auto IsRootRegionExport = [](const rvsdg::Input * input)
    {
      if (!input->region()->IsRootRegion())
      {
        return false;
      }

      if (TryGetOwnerNode<Node>(*input))
      {
        return false;
      }

      return true;
    };

    for (auto & user : output.Users())
    {
      if (!IsRootRegionExport(&user))
      {
        return false;
      }
    }

    return true;
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
      nodes.push_back(rvsdg::TryGetOwnerNode<Node>(*output));
    }
  }

  return nodes;
}

}

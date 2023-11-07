/*
 * Copyright 2010 2011 2012 2013 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <cxxabi.h>

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/substitution.hpp>

namespace jlm::rvsdg
{

/* impport */

impport::~impport()
{}

bool
impport::operator==(const port & other) const noexcept
{
	auto p = dynamic_cast<const impport*>(&other);
	return p
	    && p->type() == type()
	    && p->name() == name();
}

std::unique_ptr<port>
impport::copy() const
{
	return std::unique_ptr<port>(new impport(*this));
}

/* expport */

expport::~expport()
{}

bool
expport::operator==(const port & other) const noexcept
{
	auto p = dynamic_cast<const expport*>(&other);
	return p
	    && p->type() == type()
	    && p->name() == name();
}

std::unique_ptr<port>
expport::copy() const
{
	return std::unique_ptr<port>(new expport(*this));
}

/* graph */

graph::~graph()
{
	JLM_ASSERT(!has_active_trackers(this));

	delete root_;
}

graph::graph()
	: normalized_(false)
	, root_(new jlm::rvsdg::region(nullptr, this))
{}

std::unique_ptr<jlm::rvsdg::graph>
graph::copy() const
{
	jlm::rvsdg::substitution_map smap;
	std::unique_ptr<jlm::rvsdg::graph> graph(new jlm::rvsdg::graph());
	root()->copy(graph->root(), smap, true, true);
	return graph;
}

jlm::rvsdg::node_normal_form *
graph::node_normal_form(const std::type_info & type) noexcept
{
	auto i = node_normal_forms_.find(std::type_index(type));
	if (i != node_normal_forms_.end())
		return i.ptr();

	const auto cinfo = dynamic_cast<const abi::__si_class_type_info *>(&type);
	auto parent_normal_form = cinfo ? node_normal_form(*cinfo->__base_type) : nullptr;

	std::unique_ptr<jlm::rvsdg::node_normal_form> nf(
		jlm::rvsdg::node_normal_form::create(type, parent_normal_form, this));

	jlm::rvsdg::node_normal_form * result = nf.get();
	node_normal_forms_.insert(std::move(nf));

	return result;
}

std::vector<rvsdg::node*>
graph::ExtractTailNodes(const graph & rvsdg)
{
  auto IsOnlyExported = [](const rvsdg::output & output)
  {
    auto IsRootRegionExport = [](const rvsdg::input * input)
    {
      if (!input->region()->IsRootRegion())
      {
        return false;
      }

      if (rvsdg::node_input::node(*input))
      {
        return false;
      }

      return true;
    };

    return std::all_of(
      output.begin(),
      output.end(),
      IsRootRegionExport);
  };

  auto & rootRegion = *rvsdg.root();

  std::vector<rvsdg::node*> nodes;
  for (auto & node : rootRegion.bottom_nodes)
  {
    nodes.push_back(&node);
  }

  for (size_t n = 0; n < rootRegion.nresults(); n++)
  {
    auto output = rootRegion.result(n)->origin();
    if (IsOnlyExported(*output))
    {
      nodes.push_back(rvsdg::node_output::node(output));
    }
  }

  return nodes;
}

}

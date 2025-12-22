/*
 * Copyright 2010 2011 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_GRAPH_HPP
#define JLM_RVSDG_GRAPH_HPP

#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jlm::rvsdg
{

/**
 * Represents an import into the RVSDG of an external entity.
 */
class GraphImport : public RegionArgument
{
public:
  GraphImport(Graph & graph, std::shared_ptr<const rvsdg::Type> type, std::string name);

  [[nodiscard]] const std::string &
  Name() const noexcept
  {
    return Name_;
  }

  [[nodiscard]] std::string
  debug_string() const override;

  GraphImport &
  Copy(Region & region, StructuralInput * input) const override;

  static GraphImport &
  Create(Graph & graph, std::shared_ptr<const rvsdg::Type> type, std::string name);

private:
  std::string Name_;
};

/**
 * Represents an export from the RVSDG of an internal entity.
 */
class GraphExport : public RegionResult
{
protected:
  GraphExport(Output & origin, std::string name);

public:
  [[nodiscard]] const std::string &
  Name() const noexcept
  {
    return Name_;
  }

  [[nodiscard]] std::string
  debug_string() const override;

  GraphExport &
  Copy(Output & origin, StructuralOutput * output) const override;

  static GraphExport &
  Create(Output & origin, std::string name);

private:
  std::string Name_;
};

/**
 * Represents a Regionalized Value State Dependence Graph (RVSDG)
 */
class Graph final
{
public:
  ~Graph() noexcept;

  Graph();

  /**
   * @return The root region of the graph.
   */
  [[nodiscard]] Region &
  GetRootRegion() const noexcept
  {
    return *RootRegion_;
  }

  /**
   * @return A copy of the RVSDG.
   */
  [[nodiscard]] std::unique_ptr<Graph>
  Copy() const;

  /**
   * Remove all dead nodes in the graph.
   *
   * @see Node::IsDead()
   */
  void
  PruneNodes()
  {
    GetRootRegion().prune(true);
  }

  /**
   * Extracts all tail nodes of the RVSDG root region.
   *
   * A tail node is any node in the root region on which no other node in the root region depends
   * on. An example would be a lambda node that is not called within the RVSDG module.
   *
   * @param rvsdg The RVSDG from which to extract the tail nodes.
   * @return A vector of tail nodes.
   */
  static std::vector<Node *>
  ExtractTailNodes(const Graph & rvsdg);

private:
  std::unique_ptr<Region> RootRegion_;
};

}

#endif

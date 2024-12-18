/*
 * Copyright 2010 2011 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_GRAPH_HPP
#define JLM_RVSDG_GRAPH_HPP

#include <jlm/rvsdg/node-normal-form.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/tracker.hpp>
#include <jlm/util/common.hpp>

#include <typeindex>

namespace jlm::rvsdg
{

/**
 * Represents an import into the RVSDG of an external entity.
 */
class GraphImport : public RegionArgument
{
protected:
  GraphImport(Graph & graph, std::shared_ptr<const rvsdg::Type> type, std::string name);

public:
  [[nodiscard]] const std::string &
  Name() const noexcept
  {
    return Name_;
  }

private:
  std::string Name_;
};

/**
 * Represents an export from the RVSDG of an internal entity.
 */
class GraphExport : public RegionResult
{
protected:
  GraphExport(rvsdg::output & origin, std::string name);

public:
  [[nodiscard]] const std::string &
  Name() const noexcept
  {
    return Name_;
  }

private:
  std::string Name_;
};

class Graph
{
public:
  ~Graph();

  Graph();

  [[nodiscard]] rvsdg::Region *
  root() const noexcept
  {
    return root_;
  }

  inline void
  mark_denormalized() noexcept
  {
    normalized_ = false;
  }

  inline void
  normalize()
  {
    root()->normalize(true);
    normalized_ = true;
  }

  [[nodiscard]] std::unique_ptr<Graph>
  copy() const;

  jlm::rvsdg::node_normal_form *
  node_normal_form(const std::type_info & type) noexcept;

  inline void
  prune()
  {
    root()->prune(true);
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
  bool normalized_;
  rvsdg::Region * root_;
  jlm::rvsdg::node_normal_form_hash node_normal_forms_;
};

}

#endif

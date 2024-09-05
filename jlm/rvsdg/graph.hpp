/*
 * Copyright 2010 2011 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_GRAPH_HPP
#define JLM_RVSDG_GRAPH_HPP

#include <stdbool.h>
#include <stdlib.h>

#include <typeindex>

#include <jlm/rvsdg/node-normal-form.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/tracker.hpp>

#include <jlm/util/common.hpp>

namespace jlm::rvsdg
{

/**
 * Represents an import into the RVSDG of an external entity.
 */
class GraphImport : public argument
{
protected:
  GraphImport(rvsdg::graph & graph, std::shared_ptr<const rvsdg::type> type, std::string name);

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

class graph
{
public:
  ~graph();

  graph();

  inline jlm::rvsdg::region *
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

  std::unique_ptr<jlm::rvsdg::graph>
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
  static std::vector<rvsdg::node *>
  ExtractTailNodes(const graph & rvsdg);

private:
  bool normalized_;
  jlm::rvsdg::region * root_;
  jlm::rvsdg::node_normal_form_hash node_normal_forms_;
};

}

#endif

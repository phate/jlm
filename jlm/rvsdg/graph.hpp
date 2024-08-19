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

// FIXME: add documentation
class GraphExport : public result
{
protected:
  GraphExport(rvsdg::graph & graph, rvsdg::output & origin, std::string name);

public:
  [[nodiscard]] const std::string &
  Name() const noexcept
  {
    return Name_;
  }

private:
  std::string Name_;
};

/* expport class */

class expport : public port
{
public:
  virtual ~expport();

  expport(std::shared_ptr<const jlm::rvsdg::type> type, const std::string & name)
      : port(std::move(type)),
        name_(name)
  {}

  expport(const expport & other)
      : port(other),
        name_(other.name_)
  {}

  expport(expport && other)
      : port(other),
        name_(std::move(other.name_))
  {}

  expport &
  operator=(const expport &) = delete;

  expport &
  operator=(expport &&) = delete;

  const std::string &
  name() const noexcept
  {
    return name_;
  }

  virtual bool
  operator==(const port &) const noexcept override;

  virtual std::unique_ptr<port>
  copy() const override;

private:
  std::string name_;
};

/* graph */

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

  inline jlm::rvsdg::input *
  add_export(jlm::rvsdg::output * operand, const expport & port)
  {
    return result::create(root(), operand, nullptr, port);
  }

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

/*
 * Copyright 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_SIMPLE_NODE_HPP
#define JLM_RVSDG_SIMPLE_NODE_HPP

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/simple-normal-form.hpp>

namespace jlm::rvsdg
{

class simple_op;
class simple_input;
class simple_output;

/* simple nodes */

class simple_node : public node
{
public:
  virtual ~simple_node();

protected:
  simple_node(
      rvsdg::Region * region,
      const jlm::rvsdg::simple_op & op,
      const std::vector<jlm::rvsdg::output *> & operands);

public:
  jlm::rvsdg::simple_input *
  input(size_t index) const noexcept;

  jlm::rvsdg::simple_output *
  output(size_t index) const noexcept;

  const jlm::rvsdg::simple_op &
  operation() const noexcept;

  virtual jlm::rvsdg::node *
  copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::output *> & operands) const override;

  virtual jlm::rvsdg::node *
  copy(rvsdg::Region * region, SubstitutionMap & smap) const override;

  static inline jlm::rvsdg::simple_node *
  create(
      rvsdg::Region * region,
      const jlm::rvsdg::simple_op & op,
      const std::vector<jlm::rvsdg::output *> & operands)
  {
    return new simple_node(region, op, operands);
  }

  static inline std::vector<jlm::rvsdg::output *>
  create_normalized(
      rvsdg::Region * region,
      const jlm::rvsdg::simple_op & op,
      const std::vector<jlm::rvsdg::output *> & operands)
  {
    auto nf = static_cast<simple_normal_form *>(region->graph()->node_normal_form(typeid(op)));
    return nf->normalized_create(region, op, operands);
  }
};

/* inputs */

class simple_input final : public node_input
{
  friend class jlm::rvsdg::output;

public:
  virtual ~simple_input() noexcept;

  simple_input(
      simple_node * node,
      jlm::rvsdg::output * origin,
      std::shared_ptr<const rvsdg::Type> type);

public:
  simple_node *
  node() const noexcept
  {
    return static_cast<simple_node *>(node_input::node());
  }
};

/* outputs */

class simple_output final : public node_output
{
  friend class jlm::rvsdg::simple_input;

public:
  virtual ~simple_output() noexcept;

  simple_output(jlm::rvsdg::simple_node * node, std::shared_ptr<const rvsdg::Type> type);

public:
  simple_node *
  node() const noexcept
  {
    return static_cast<simple_node *>(node_output::node());
  }
};

/* simple node method definitions */

inline jlm::rvsdg::simple_input *
simple_node::input(size_t index) const noexcept
{
  return static_cast<simple_input *>(node::input(index));
}

inline jlm::rvsdg::simple_output *
simple_node::output(size_t index) const noexcept
{
  return static_cast<simple_output *>(node::output(index));
}

inline const jlm::rvsdg::simple_op &
simple_node::operation() const noexcept
{
  return *static_cast<const simple_op *>(&node::operation());
}

}

#endif

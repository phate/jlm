/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_STATEMUX_HPP
#define JLM_RVSDG_STATEMUX_HPP

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/simple-normal-form.hpp>

namespace jlm::rvsdg
{

/* mux normal form */

class mux_normal_form final : public simple_normal_form
{
public:
  virtual ~mux_normal_form() noexcept;

  mux_normal_form(
      const std::type_info & opclass,
      jlm::rvsdg::node_normal_form * parent,
      jlm::rvsdg::graph * graph) noexcept;

  virtual bool
  normalize_node(jlm::rvsdg::node * node) const override;

  virtual std::vector<jlm::rvsdg::output *>
  normalized_create(
      jlm::rvsdg::region * region,
      const jlm::rvsdg::simple_op & op,
      const std::vector<jlm::rvsdg::output *> & arguments) const override;

  virtual void
  set_mux_mux_reducible(bool enable);

  virtual void
  set_multiple_origin_reducible(bool enable);

  inline bool
  get_mux_mux_reducible() const noexcept
  {
    return enable_mux_mux_;
  }

  inline bool
  get_multiple_origin_reducible() const noexcept
  {
    return enable_multiple_origin_;
  }

private:
  bool enable_mux_mux_;
  bool enable_multiple_origin_;
};

/* mux operation */

class mux_op final : public simple_op
{
public:
  virtual ~mux_op() noexcept;

  inline mux_op(std::shared_ptr<const statetype> type, size_t narguments, size_t nresults)
      : simple_op({ narguments, type }, { nresults, type })
  {}

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static jlm::rvsdg::mux_normal_form *
  normal_form(jlm::rvsdg::graph * graph) noexcept
  {
    return static_cast<jlm::rvsdg::mux_normal_form *>(graph->node_normal_form(typeid(mux_op)));
  }
};

static inline bool
is_mux_op(const jlm::rvsdg::operation & op)
{
  return dynamic_cast<const jlm::rvsdg::mux_op *>(&op) != nullptr;
}

static inline std::vector<jlm::rvsdg::output *>
create_state_mux(
    std::shared_ptr<const jlm::rvsdg::type> type,
    const std::vector<jlm::rvsdg::output *> & operands,
    size_t nresults)
{
  if (operands.empty())
    throw jlm::util::error("Insufficient number of operands.");

  auto st = std::dynamic_pointer_cast<const jlm::rvsdg::statetype>(type);
  if (!st)
    throw jlm::util::error("Expected state type.");

  auto region = operands.front()->region();
  jlm::rvsdg::mux_op op(std::move(st), operands.size(), nresults);
  return simple_node::create_normalized(region, op, operands);
}

static inline jlm::rvsdg::output *
create_state_merge(
    std::shared_ptr<const jlm::rvsdg::type> type,
    const std::vector<jlm::rvsdg::output *> & operands)
{
  return create_state_mux(std::move(type), operands, 1)[0];
}

static inline std::vector<jlm::rvsdg::output *>
create_state_split(
    std::shared_ptr<const jlm::rvsdg::type> type,
    jlm::rvsdg::output * operand,
    size_t nresults)
{
  return create_state_mux(std::move(type), { operand }, nresults);
}

}

#endif

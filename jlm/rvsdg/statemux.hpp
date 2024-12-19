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

#include <optional>

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
      Graph * graph) noexcept;

  virtual bool
  normalize_node(Node * node) const override;

  virtual std::vector<jlm::rvsdg::output *>
  normalized_create(
      rvsdg::Region * region,
      const SimpleOperation & op,
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

class mux_op final : public SimpleOperation
{
public:
  virtual ~mux_op() noexcept;

  inline mux_op(std::shared_ptr<const StateType> type, size_t narguments, size_t nresults)
      : SimpleOperation({ narguments, type }, { nresults, type })
  {}

  virtual bool
  operator==(const Operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static jlm::rvsdg::mux_normal_form *
  normal_form(Graph * graph) noexcept
  {
    return static_cast<jlm::rvsdg::mux_normal_form *>(graph->node_normal_form(typeid(mux_op)));
  }
};

static inline bool
is_mux_op(const Operation & op)
{
  return dynamic_cast<const jlm::rvsdg::mux_op *>(&op) != nullptr;
}

static inline std::vector<jlm::rvsdg::output *>
create_state_mux(
    std::shared_ptr<const jlm::rvsdg::Type> type,
    const std::vector<jlm::rvsdg::output *> & operands,
    size_t nresults)
{
  if (operands.empty())
    throw jlm::util::error("Insufficient number of operands.");

  auto st = std::dynamic_pointer_cast<const StateType>(type);
  if (!st)
    throw jlm::util::error("Expected state type.");

  auto region = operands.front()->region();
  jlm::rvsdg::mux_op op(std::move(st), operands.size(), nresults);
  return simple_node::create_normalized(region, op, operands);
}

static inline jlm::rvsdg::output *
create_state_merge(
    std::shared_ptr<const jlm::rvsdg::Type> type,
    const std::vector<jlm::rvsdg::output *> & operands)
{
  return create_state_mux(std::move(type), operands, 1)[0];
}

static inline std::vector<jlm::rvsdg::output *>
create_state_split(
    std::shared_ptr<const jlm::rvsdg::Type> type,
    jlm::rvsdg::output * operand,
    size_t nresults)
{
  return create_state_mux(std::move(type), { operand }, nresults);
}

/**
 * \brief Merges multiple mux operations into a single operation.
 *
 * so1 = mux_op si1 si2
 * so2 = mux_op si3 si4
 * so3 = mux_op so1 so2
 * =>
 * so3 = mux_op si1 si2 si3 si4
 *
 * @param operation The mux operation on which the transformation is performed.
 * @param operands The operands of the mux node.
 *
 * @return If the normalization could be applied, then the results of the mux operation after
 * the transformation. Otherwise, std::nullopt.
 */
std::optional<std::vector<rvsdg::output *>>
NormalizeMuxMux(const mux_op & operation, const std::vector<rvsdg::output *> & operands);

/**
 * \brief Remove duplicated operands
 *
 * so1 = mux_op si1 si1 si2 si1 si3
 * =>
 * so1 = mux_op si1 si2 si3
 *
 * @param operation The mux operation on which the transformation is performed.
 * @param operands The operands of the mux node.
 *
 * @return If the normalization could be applied, then the results of the mux operation after
 * the transformation. Otherwise, std::nullopt.
 */
std::optional<std::vector<rvsdg::output *>>
NormalizeMuxDuplicateOperands(
    const mux_op & operation,
    const std::vector<rvsdg::output *> & operands);

}

#endif

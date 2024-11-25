/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/statemux.hpp>
#include <jlm/util/HashSet.hpp>

namespace jlm::rvsdg
{

/* mux operator */

mux_op::~mux_op() noexcept
{}

bool
mux_op::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const mux_op *>(&other);
  return op && op->narguments() == narguments() && op->nresults() == nresults()
      && op->result(0) == result(0);
}

std::string
mux_op::debug_string() const
{
  return "STATEMUX";
}

std::unique_ptr<jlm::rvsdg::operation>
mux_op::copy() const
{
  return std::unique_ptr<jlm::rvsdg::operation>(new mux_op(*this));
}

MuxMuxReduction::~MuxMuxReduction() noexcept = default;

bool
MuxMuxReduction::IsApplicable(const mux_op & operation, const std::vector<output *> & operands)
{
  ResetState();

  const util::HashSet<output *> operandSet(operands.begin(), operands.end());

  for (const auto & operand : operands)
  {
    const auto node = output::GetNode(*operand);
    if (!node || !is_mux_op(node->operation()))
      continue;

    size_t n;
    for (n = 0; n < node->noutputs(); n++)
    {
      if (const auto output = node->output(n);
          !operandSet.Contains(output) || output->nusers() != 1)
        break;
    }
    if (n == node->noutputs())
    {
      MuxNode_ = node;
      return true;
    }
  }

  return false;
}

std::vector<output *>
MuxMuxReduction::ApplyNormalization(const mux_op & operation, const std::vector<output *> & operands)
{
  JLM_ASSERT(is_mux_op(MuxNode_->operation()));

  bool reduced = false;
  std::vector<output *> newOperands;
  for (const auto & operand : operands)
  {
    if (output::GetNode(*operand) == MuxNode_ && !reduced)
    {
      reduced = true;
      auto tmp = rvsdg::operands(MuxNode_);
      newOperands.insert(newOperands.end(), tmp.begin(), tmp.end());
      continue;
    }

    if (output::GetNode(*operand) != MuxNode_)
      newOperands.push_back(operand);
  }

  return create_state_mux(operation.result(0), newOperands, operation.nresults());
}

MuxDuplicateOriginReduction::~MuxDuplicateOriginReduction() noexcept = default;

bool
MuxDuplicateOriginReduction::IsApplicable(
    const mux_op & operation,
    const std::vector<output *> & operands)
{
  const util::HashSet<output *> set(operands.begin(), operands.end());
  return set.Size() != operands.size();
}

std::vector<output *>
MuxDuplicateOriginReduction::ApplyNormalization(
    const mux_op & operation,
    const std::vector<output *> & operands)
{
  const util::HashSet<output *> set(operands.begin(), operands.end());
  return create_state_mux(
      operation.result(0),
      { set.Items().begin(), set.Items().end() },
      operation.nresults());
}

static MuxMuxReduction muxMuxReduction;
static MuxDuplicateOriginReduction muxDuplicateOriginReduction;

mux_normal_form::~mux_normal_form() noexcept
{}

mux_normal_form::mux_normal_form(
    const std::type_info & opclass,
    jlm::rvsdg::node_normal_form * parent,
    jlm::rvsdg::graph * graph) noexcept
    : simple_normal_form(opclass, parent, graph),
      enable_mux_mux_(false),
      enable_multiple_origin_(false)
{
  if (auto p = dynamic_cast<const mux_normal_form *>(parent))
    enable_mux_mux_ = p->enable_mux_mux_;
}

bool
mux_normal_form::normalize_node(jlm::rvsdg::node * node) const
{
  JLM_ASSERT(dynamic_cast<const jlm::rvsdg::mux_op *>(&node->operation()));
  const auto & operation = *util::AssertedCast<const mux_op>(&node->operation());
  const auto operands = rvsdg::operands(node);

  if (!get_mutable())
    return true;

  if (get_mux_mux_reducible() && muxMuxReduction.IsApplicable(operation, operands))
  {
    divert_users(node, muxMuxReduction.ApplyNormalization(operation, operands));
    remove(node);
    return false;
  }

  if (get_multiple_origin_reducible()
      && muxDuplicateOriginReduction.IsApplicable(operation, operands))
  {
    divert_users(node, muxDuplicateOriginReduction.ApplyNormalization(operation, operands));
    remove(node);
    return false;
  }

  return simple_normal_form::normalize_node(node);
}

std::vector<jlm::rvsdg::output *>
mux_normal_form::normalized_create(
    Region * region,
    const simple_op & op,
    const std::vector<output *> & operands) const
{
  JLM_ASSERT(dynamic_cast<const jlm::rvsdg::mux_op *>(&op));
  const auto & operation = *util::AssertedCast<const mux_op>(&op);

  if (!get_mutable())
    return simple_normal_form::normalized_create(region, op, operands);

  if (get_mux_mux_reducible() && muxMuxReduction.IsApplicable(operation, operands))
    return muxMuxReduction.ApplyNormalization(operation, operands);

  if (get_multiple_origin_reducible()
      && muxDuplicateOriginReduction.IsApplicable(operation, operands))
    return muxDuplicateOriginReduction.ApplyNormalization(operation, operands);

  return simple_normal_form::normalized_create(region, op, operands);
}

void
mux_normal_form::set_mux_mux_reducible(bool enable)
{
  if (get_mux_mux_reducible() == enable)
    return;

  children_set<mux_normal_form, &mux_normal_form::set_mux_mux_reducible>(enable);

  enable_mux_mux_ = enable;
  if (get_mutable() && enable)
    graph()->mark_denormalized();
}

void
mux_normal_form::set_multiple_origin_reducible(bool enable)
{
  if (get_multiple_origin_reducible() == enable)
    return;

  children_set<mux_normal_form, &mux_normal_form::set_multiple_origin_reducible>(enable);

  enable_multiple_origin_ = enable;
  if (get_mutable() && enable)
    graph()->mark_denormalized();
}

}

namespace
{

static jlm::rvsdg::node_normal_form *
create_mux_normal_form(
    const std::type_info & opclass,
    jlm::rvsdg::node_normal_form * parent,
    jlm::rvsdg::graph * graph)
{
  return new jlm::rvsdg::mux_normal_form(opclass, parent, graph);
}

static void __attribute__((constructor))
register_node_normal_form(void)
{
  jlm::rvsdg::node_normal_form::register_factory(
      typeid(jlm::rvsdg::mux_op),
      create_mux_normal_form);
}

}

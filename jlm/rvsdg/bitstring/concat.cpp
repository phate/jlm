/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/concat.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/bitstring/slice.hpp>
#include <jlm/rvsdg/reduction-helpers.hpp>

namespace jlm::rvsdg
{

jlm::rvsdg::output *
bitconcat(const std::vector<jlm::rvsdg::output *> & operands)
{
  std::vector<std::shared_ptr<const jlm::rvsdg::bittype>> types;
  for (const auto operand : operands)
    types.push_back(std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(operand->Type()));

  auto region = operands[0]->region();
  jlm::rvsdg::bitconcat_op op(std::move(types));
  return jlm::rvsdg::SimpleNode::create_normalized(
      region,
      op,
      { operands.begin(), operands.end() })[0];
}

namespace
{

jlm::rvsdg::output *
concat_reduce_arg_pair(jlm::rvsdg::output * arg1, jlm::rvsdg::output * arg2)
{
  auto node1 = output::GetNode(*arg1);
  auto node2 = output::GetNode(*arg2);
  if (!node1 || !node2)
    return nullptr;

  auto arg1_constant = dynamic_cast<const bitconstant_op *>(&node1->GetOperation());
  auto arg2_constant = dynamic_cast<const bitconstant_op *>(&node2->GetOperation());
  if (arg1_constant && arg2_constant)
  {
    size_t nbits = arg1_constant->value().nbits() + arg2_constant->value().nbits();
    std::vector<char> bits(nbits);
    memcpy(&bits[0], &arg1_constant->value()[0], arg1_constant->value().nbits());
    memcpy(
        &bits[0] + arg1_constant->value().nbits(),
        &arg2_constant->value()[0],
        arg2_constant->value().nbits());

    std::string s(&bits[0], nbits);
    return create_bitconstant(node1->region(), s.c_str());
  }

  auto arg1_slice = dynamic_cast<const bitslice_op *>(&node1->GetOperation());
  auto arg2_slice = dynamic_cast<const bitslice_op *>(&node2->GetOperation());
  if (arg1_slice && arg2_slice && arg1_slice->high() == arg2_slice->low()
      && node1->input(0)->origin() == node2->input(0)->origin())
  {
    /* FIXME: support sign bit */
    return jlm::rvsdg::bitslice(node1->input(0)->origin(), arg1_slice->low(), arg2_slice->high());
  }

  return nullptr;
}

std::vector<std::shared_ptr<const bittype>>
types_from_arguments(const std::vector<jlm::rvsdg::output *> & args)
{
  std::vector<std::shared_ptr<const bittype>> types;
  for (const auto arg : args)
  {
    types.push_back(std::dynamic_pointer_cast<const bittype>(arg->Type()));
  }
  return types;
}

}

class concat_normal_form final : public simple_normal_form
{
public:
  virtual ~concat_normal_form() noexcept;

  concat_normal_form(jlm::rvsdg::node_normal_form * parent, Graph * graph)
      : simple_normal_form(typeid(bitconcat_op), parent, graph),
        enable_reducible_(true),
        enable_flatten_(true)
  {}

  virtual bool
  normalize_node(Node * node) const override
  {
    if (!get_mutable())
    {
      return true;
    }

    auto args = operands(node);
    std::vector<jlm::rvsdg::output *> new_args;

    /* possibly expand associative */
    if (get_flatten())
    {
      new_args = base::detail::associative_flatten(
          args,
          [](jlm::rvsdg::output * arg)
          {
            // FIXME: switch to comparing operator, not just typeid, after
            // converting "concat" to not be a binary operator anymore
            return is<bitconcat_op>(output::GetNode(*arg));
          });
    }
    else
    {
      new_args = args;
    }

    if (get_reducible())
    {
      new_args = base::detail::pairwise_reduce(std::move(new_args), concat_reduce_arg_pair);

      if (new_args.size() == 1)
      {
        divert_users(node, new_args);
        remove(node);
        return false;
      }
    }

    if (args != new_args)
    {
      bitconcat_op op(types_from_arguments(new_args));
      divert_users(node, SimpleNode::create_normalized(node->region(), op, new_args));
      remove(node);
      return false;
    }

    return simple_normal_form::normalize_node(node);
  }

  virtual std::vector<jlm::rvsdg::output *>
  normalized_create(
      rvsdg::Region * region,
      const SimpleOperation &,
      const std::vector<jlm::rvsdg::output *> & arguments) const override
  {
    std::vector<jlm::rvsdg::output *> new_args;

    /* possibly expand associative */
    if (get_mutable() && get_flatten())
    {
      new_args = base::detail::associative_flatten(
          arguments,
          [](jlm::rvsdg::output * arg)
          {
            // FIXME: switch to comparing operator, not just typeid, after
            // converting "concat" to not be a binary operator anymore
            return is<bitconcat_op>(output::GetNode(*arg));
          });
    }
    else
    {
      new_args = arguments;
    }

    if (get_mutable() && get_reducible())
    {
      new_args = base::detail::pairwise_reduce(std::move(new_args), concat_reduce_arg_pair);
      if (new_args.size() == 1)
        return new_args;
    }

    bitconcat_op new_op(types_from_arguments(new_args));
    return simple_normal_form::normalized_create(region, new_op, new_args);
  }

  virtual void
  set_reducible(bool enable)
  {
    if (get_reducible() == enable)
    {
      return;
    }

    children_set<concat_normal_form, &concat_normal_form::set_reducible>(enable);

    enable_reducible_ = enable;
    if (get_mutable() && enable)
      graph()->MarkDenormalized();
  }

  inline bool
  get_reducible() const noexcept
  {
    return enable_reducible_;
  }

  virtual void
  set_flatten(bool enable)
  {
    if (get_flatten() == enable)
    {
      return;
    }

    children_set<concat_normal_form, &concat_normal_form::set_flatten>(enable);

    enable_flatten_ = enable;
    if (get_mutable() && enable)
      graph()->MarkDenormalized();
  }

  inline bool
  get_flatten() const noexcept
  {
    return enable_flatten_;
  }

private:
  bool enable_reducible_;
  bool enable_flatten_;
};

concat_normal_form::~concat_normal_form() noexcept
{}

static node_normal_form *
get_default_normal_form(
    const std::type_info &,
    jlm::rvsdg::node_normal_form * parent,
    Graph * graph)
{
  return new concat_normal_form(parent, graph);
}

static void __attribute__((constructor))
register_node_normal_form()
{
  jlm::rvsdg::node_normal_form::register_factory(
      typeid(jlm::rvsdg::bitconcat_op),
      get_default_normal_form);
}

std::shared_ptr<const bittype>
bitconcat_op::aggregate_arguments(
    const std::vector<std::shared_ptr<const bittype>> & types) noexcept
{
  size_t total = 0;
  for (const auto & t : types)
  {
    total += t->nbits();
  }
  return bittype::Create(total);
}

bitconcat_op::~bitconcat_op() noexcept
{}

bool
bitconcat_op::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const jlm::rvsdg::bitconcat_op *>(&other);
  if (!op || op->narguments() != narguments())
    return false;

  for (size_t n = 0; n < narguments(); n++)
  {
    if (op->argument(n) != argument(n))
      return false;
  }

  return true;
}

binop_reduction_path_t
bitconcat_op::can_reduce_operand_pair(
    const jlm::rvsdg::output * arg1,
    const jlm::rvsdg::output * arg2) const noexcept
{
  auto node1 = output::GetNode(*arg1);
  auto node2 = output::GetNode(*arg2);

  if (!node1 || !node2)
    return binop_reduction_none;

  auto arg1_constant = is<bitconstant_op>(node1);
  auto arg2_constant = is<bitconstant_op>(node2);

  if (arg1_constant && arg2_constant)
  {
    return binop_reduction_constants;
  }

  auto arg1_slice = dynamic_cast<const bitslice_op *>(&node1->GetOperation());
  auto arg2_slice = dynamic_cast<const bitslice_op *>(&node2->GetOperation());

  if (arg1_slice && arg2_slice)
  {
    auto origin1 = node1->input(0)->origin();
    auto origin2 = node2->input(0)->origin();

    if (origin1 == origin2 && arg1_slice->high() == arg2_slice->low())
    {
      return binop_reduction_merge;
    }

    /* FIXME: support sign bit */
  }

  return binop_reduction_none;
}

jlm::rvsdg::output *
bitconcat_op::reduce_operand_pair(
    binop_reduction_path_t path,
    jlm::rvsdg::output * arg1,
    jlm::rvsdg::output * arg2) const
{
  auto node1 = static_cast<node_output *>(arg1)->node();
  auto node2 = static_cast<node_output *>(arg2)->node();

  if (path == binop_reduction_constants)
  {
    auto & arg1_constant = static_cast<const bitconstant_op &>(node1->GetOperation());
    auto & arg2_constant = static_cast<const bitconstant_op &>(node2->GetOperation());

    size_t nbits = arg1_constant.value().nbits() + arg2_constant.value().nbits();
    std::vector<char> bits(nbits);
    memcpy(&bits[0], &arg1_constant.value()[0], arg1_constant.value().nbits());
    memcpy(
        &bits[0] + arg1_constant.value().nbits(),
        &arg2_constant.value()[0],
        arg2_constant.value().nbits());

    return create_bitconstant(arg1->region(), &bits[0]);
  }

  if (path == binop_reduction_merge)
  {
    auto arg1_slice = static_cast<const bitslice_op *>(&node1->GetOperation());
    auto arg2_slice = static_cast<const bitslice_op *>(&node2->GetOperation());
    return jlm::rvsdg::bitslice(node1->input(0)->origin(), arg1_slice->low(), arg2_slice->high());

    /* FIXME: support sign bit */
  }

  return NULL;
}

enum BinaryOperation::flags
bitconcat_op::flags() const noexcept
{
  return BinaryOperation::flags::associative;
}

std::string
bitconcat_op::debug_string() const
{
  return "BITCONCAT";
}

std::unique_ptr<Operation>
bitconcat_op::copy() const
{
  return std::make_unique<bitconcat_op>(*this);
}

static std::vector<std::shared_ptr<const bittype>>
GetTypesFromOperands(const std::vector<rvsdg::output *> & args)
{
  std::vector<std::shared_ptr<const bittype>> types;
  for (const auto arg : args)
  {
    types.push_back(std::dynamic_pointer_cast<const bittype>(arg->Type()));
  }
  return types;
}

std::optional<std::vector<rvsdg::output *>>
FlattenBitConcatOperation(const bitconcat_op &, const std::vector<rvsdg::output *> & operands)
{
  JLM_ASSERT(!operands.empty());

  const auto newOperands = base::detail::associative_flatten(
      operands,
      [](jlm::rvsdg::output * arg)
      {
        // FIXME: switch to comparing operator, not just typeid, after
        // converting "concat" to not be a binary operator anymore
        return is<bitconcat_op>(output::GetNode(*arg));
      });

  if (operands == newOperands)
  {
    JLM_ASSERT(newOperands.size() == 2);
    return std::nullopt;
  }

  JLM_ASSERT(newOperands.size() > 2);
  return outputs(&CreateOpNode<bitconcat_op>(newOperands, GetTypesFromOperands(newOperands)));
}

}

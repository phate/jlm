/*
 * Copyright 2010 2011 2012 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/node-normal-form.hpp>
#include <jlm/rvsdg/notifiers.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/theta.hpp>

namespace jlm::rvsdg
{

/* input */

input::~input() noexcept
{
  origin()->remove_user(this);
}

input::input(
    jlm::rvsdg::output * origin,
    rvsdg::Region * region,
    std::shared_ptr<const rvsdg::Type> type)
    : index_(0),
      origin_(origin),
      region_(region),
      Type_(std::move(type))
{
  if (region != origin->region())
    throw jlm::util::error("Invalid operand region.");

  if (*Type() != origin->type())
    throw jlm::util::type_error(Type()->debug_string(), origin->type().debug_string());

  origin->add_user(this);
}

std::string
input::debug_string() const
{
  return jlm::util::strfmt(index());
}

void
input::divert_to(jlm::rvsdg::output * new_origin)
{
  if (origin() == new_origin)
    return;

  if (type() != new_origin->type())
    throw jlm::util::type_error(type().debug_string(), new_origin->type().debug_string());

  if (region() != new_origin->region())
    throw jlm::util::error("Invalid operand region.");

  auto old_origin = origin();
  old_origin->remove_user(this);
  this->origin_ = new_origin;
  new_origin->add_user(this);

  if (is<node_input>(*this))
    static_cast<node_input *>(this)->node()->recompute_depth();

  region()->graph()->mark_denormalized();
  on_input_change(this, old_origin, new_origin);
}

rvsdg::node *
input::GetNode(const rvsdg::input & input) noexcept
{
  auto nodeInput = dynamic_cast<const rvsdg::node_input *>(&input);
  return nodeInput ? nodeInput->node() : nullptr;
}

/* output */

output::~output() noexcept
{
  JLM_ASSERT(nusers() == 0);
}

output::output(rvsdg::Region * region, std::shared_ptr<const rvsdg::Type> type)
    : index_(0),
      region_(region),
      Type_(std::move(type))
{}

std::string
output::debug_string() const
{
  return jlm::util::strfmt(index());
}

rvsdg::node *
output::GetNode(const rvsdg::output & output) noexcept
{
  auto nodeOutput = dynamic_cast<const rvsdg::node_output *>(&output);
  return nodeOutput ? nodeOutput->node() : nullptr;
}

void
output::remove_user(jlm::rvsdg::input * user)
{
  JLM_ASSERT(users_.find(user) != users_.end());

  users_.erase(user);

  if (auto node = output::GetNode(*this))
  {
    if (!node->has_users())
    {
      JLM_ASSERT(region()->AddBottomNode(*node));
    }
  }
}

void
output::add_user(jlm::rvsdg::input * user)
{
  JLM_ASSERT(users_.find(user) == users_.end());

  if (auto node = output::GetNode(*this))
  {
    if (!node->has_users())
    {
      JLM_ASSERT(region()->RemoveBottomNode(*node));
    }
  }
  users_.insert(user);
}

}

jlm::rvsdg::node_normal_form *
node_get_default_normal_form_(
    const std::type_info & operator_class,
    jlm::rvsdg::node_normal_form * parent,
    jlm::rvsdg::graph * graph)
{
  return new jlm::rvsdg::node_normal_form(operator_class, parent, graph);
}

static void __attribute__((constructor))
register_node_normal_form(void)
{
  jlm::rvsdg::node_normal_form::register_factory(
      typeid(jlm::rvsdg::operation),
      node_get_default_normal_form_);
}

namespace jlm::rvsdg
{

/* node_input  class */

node_input::node_input(
    jlm::rvsdg::output * origin,
    jlm::rvsdg::node * node,
    std::shared_ptr<const rvsdg::Type> type)
    : jlm::rvsdg::input(origin, node->region(), std::move(type)),
      node_(node)
{}

/* node_output class */

node_output::node_output(jlm::rvsdg::node * node, std::shared_ptr<const rvsdg::Type> type)
    : jlm::rvsdg::output(node->region(), std::move(type)),
      node_(node)
{}

/* node class */

node::node(std::unique_ptr<jlm::rvsdg::operation> op, rvsdg::Region * region)
    : depth_(0),
      graph_(region->graph()),
      region_(region),
      operation_(std::move(op))
{
  JLM_ASSERT(region->AddBottomNode(*this));
  region->top_nodes.push_back(this);
  region->nodes.push_back(this);
}

node::~node()
{
  outputs_.clear();
  JLM_ASSERT(region()->RemoveBottomNode(*this));

  if (ninputs() == 0)
    region()->top_nodes.erase(this);
  inputs_.clear();

  region()->nodes.erase(this);
}

node_input *
node::add_input(std::unique_ptr<node_input> input)
{
  auto producer = output::GetNode(*input->origin());

  if (ninputs() == 0)
  {
    JLM_ASSERT(depth() == 0);
    region()->top_nodes.erase(this);
  }

  input->index_ = ninputs();
  inputs_.push_back(std::move(input));

  auto new_depth = producer ? producer->depth() + 1 : 0;
  if (new_depth > depth())
    recompute_depth();

  return this->input(ninputs() - 1);
}

void
node::RemoveInput(size_t index)
{
  JLM_ASSERT(index < ninputs());
  auto producer = output::GetNode(*input(index)->origin());

  /* remove input */
  for (size_t n = index; n < ninputs() - 1; n++)
  {
    inputs_[n] = std::move(inputs_[n + 1]);
    inputs_[n]->index_ = n;
  }
  inputs_.pop_back();

  /* recompute depth */
  if (producer)
  {
    auto pdepth = producer->depth();
    JLM_ASSERT(pdepth < depth());
    if (pdepth != depth() - 1)
      return;
  }
  recompute_depth();

  /* add to region's top nodes */
  if (ninputs() == 0)
  {
    JLM_ASSERT(depth() == 0);
    region()->top_nodes.push_back(this);
  }
}

void
node::RemoveOutput(size_t index)
{
  JLM_ASSERT(index < noutputs());

  for (size_t n = index; n < noutputs() - 1; n++)
  {
    outputs_[n] = std::move(outputs_[n + 1]);
    outputs_[n]->index_ = n;
  }
  outputs_.pop_back();
}

void
node::recompute_depth() noexcept
{
  /*
    FIXME: This function is inefficient, as it can visit the
    node's successors multiple times. Optimally, we would like
    to visit the node's successors in top down order to ensure
    that each node is only visited once.
  */
  size_t new_depth = 0;
  for (size_t n = 0; n < ninputs(); n++)
  {
    auto producer = output::GetNode(*input(n)->origin());
    new_depth = std::max(new_depth, producer ? producer->depth() + 1 : 0);
  }
  if (new_depth == depth())
    return;

  size_t old_depth = depth();
  depth_ = new_depth;
  on_node_depth_change(this, old_depth);

  for (size_t n = 0; n < noutputs(); n++)
  {
    for (auto user : *(output(n)))
    {
      if (!is<node_input>(*user))
        continue;

      auto node = static_cast<node_input *>(user)->node();
      node->recompute_depth();
    }
  }
}

jlm::rvsdg::node *
node::copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::output *> & operands) const
{
  SubstitutionMap smap;

  size_t noperands = std::min(operands.size(), ninputs());
  for (size_t n = 0; n < noperands; n++)
    smap.insert(input(n)->origin(), operands[n]);

  return copy(region, smap);
}

jlm::rvsdg::node *
producer(const jlm::rvsdg::output * output) noexcept
{
  if (auto node = output::GetNode(*output))
    return node;

  JLM_ASSERT(dynamic_cast<const RegionArgument *>(output));
  auto argument = static_cast<const RegionArgument *>(output);

  if (!argument->input())
    return nullptr;

  if (is<ThetaOperation>(argument->region()->node())
      && (argument->region()->result(argument->index() + 1)->origin() != argument))
    return nullptr;

  return producer(argument->input()->origin());
}

bool
normalize(jlm::rvsdg::node * node)
{
  const auto & op = node->operation();
  auto nf = node->graph()->node_normal_form(typeid(op));
  return nf->normalize_node(node);
}

}

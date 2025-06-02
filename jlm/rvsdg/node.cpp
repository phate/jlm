/*
 * Copyright 2010 2011 2012 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/notifiers.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/theta.hpp>

namespace jlm::rvsdg
{

Input::~Input() noexcept
{
  origin()->remove_user(this);
}

Input::Input(
    jlm::rvsdg::Output * origin,
    rvsdg::Region * region,
    std::shared_ptr<const rvsdg::Type> type)
    : index_(0),
      origin_(origin),
      region_(region),
      Type_(std::move(type))
{
  if (region != origin->region())
    throw jlm::util::error("Invalid operand region.");

  if (*Type() != *origin->Type())
    throw util::TypeError(Type()->debug_string(), origin->Type()->debug_string());

  origin->add_user(this);
}

std::string
Input::debug_string() const
{
  return jlm::util::strfmt("i", index());
}

void
Input::divert_to(jlm::rvsdg::Output * new_origin)
{
  if (origin() == new_origin)
    return;

  if (*Type() != *new_origin->Type())
    throw jlm::util::TypeError(Type()->debug_string(), new_origin->Type()->debug_string());

  if (region() != new_origin->region())
    throw jlm::util::error("Invalid operand region.");

  auto old_origin = origin();
  old_origin->remove_user(this);
  this->origin_ = new_origin;
  new_origin->add_user(this);

  if (auto node = TryGetOwnerNode<Node>(*this))
    node->recompute_depth();

  on_input_change(this, old_origin, new_origin);
}

Output::~Output() noexcept
{
  JLM_ASSERT(nusers() == 0);
}

Output::Output(rvsdg::Region * region, std::shared_ptr<const rvsdg::Type> type)
    : index_(0),
      region_(region),
      Type_(std::move(type))
{}

std::string
Output::debug_string() const
{
  return jlm::util::strfmt("o", index());
}

void
Output::remove_user(jlm::rvsdg::Input * user)
{
  JLM_ASSERT(users_.find(user) != users_.end());

  users_.erase(user);

  if (auto node = TryGetOwnerNode<Node>(*this))
  {
    if (!node->has_users())
    {
      bool wasAdded = region()->AddBottomNode(*node);
      JLM_ASSERT(wasAdded);
    }
  }
}

void
Output::add_user(jlm::rvsdg::Input * user)
{
  JLM_ASSERT(users_.find(user) == users_.end());

  if (auto node = TryGetOwnerNode<Node>(*this))
  {
    if (!node->has_users())
    {
      bool wasRemoved = region()->RemoveBottomNode(*node);
      JLM_ASSERT(wasRemoved);
    }
  }
  users_.insert(user);
}

node_input::node_input(
    jlm::rvsdg::Output * origin,
    Node * node,
    std::shared_ptr<const rvsdg::Type> type)
    : jlm::rvsdg::Input(origin, node->region(), std::move(type)),
      node_(node)
{}

[[nodiscard]] std::variant<Node *, Region *>
node_input::GetOwner() const noexcept
{
  return node_;
}

/* node_output class */

node_output::node_output(Node * node, std::shared_ptr<const rvsdg::Type> type)
    : jlm::rvsdg::Output(node->region(), std::move(type)),
      node_(node)
{}

[[nodiscard]] std::variant<Node *, Region *>
node_output::GetOwner() const noexcept
{
  return node_;
}

/* node class */

Node::Node(Region * region)
    : depth_(0),
      graph_(region->graph()),
      region_(region)
{
  bool wasAdded = region->AddBottomNode(*this);
  JLM_ASSERT(wasAdded);
  wasAdded = region->AddTopNode(*this);
  JLM_ASSERT(wasAdded);
  wasAdded = region->AddNode(*this);
  JLM_ASSERT(wasAdded);
}

Node::~Node()
{
  outputs_.clear();
  bool wasRemoved = region()->RemoveBottomNode(*this);
  JLM_ASSERT(wasRemoved);

  if (ninputs() == 0)
  {
    wasRemoved = region()->RemoveTopNode(*this);
    JLM_ASSERT(wasRemoved);
  }
  inputs_.clear();

  wasRemoved = region()->RemoveNode(*this);
  JLM_ASSERT(wasRemoved);
}

node_input *
Node::add_input(std::unique_ptr<node_input> input)
{
  auto producer = rvsdg::TryGetOwnerNode<Node>(*input->origin());

  if (ninputs() == 0)
  {
    JLM_ASSERT(depth() == 0);
    const auto wasRemoved = region()->RemoveTopNode(*this);
    JLM_ASSERT(wasRemoved);
  }

  input->index_ = ninputs();
  inputs_.push_back(std::move(input));

  auto new_depth = producer ? producer->depth() + 1 : 0;
  if (new_depth > depth())
    recompute_depth();

  return this->input(ninputs() - 1);
}

void
Node::RemoveInput(size_t index)
{
  JLM_ASSERT(index < ninputs());
  auto producer = rvsdg::TryGetOwnerNode<Node>(*input(index)->origin());

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
    const auto wasAdded = region()->AddTopNode(*this);
    JLM_ASSERT(wasAdded);
  }
}

void
Node::RemoveOutput(size_t index)
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
Node::recompute_depth() noexcept
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
    auto producer = rvsdg::TryGetOwnerNode<Node>(*input(n)->origin());
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
      if (auto node = TryGetOwnerNode<Node>(*user))
      {
        node->recompute_depth();
      }
    }
  }
}

Node *
Node::copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::Output *> & operands) const
{
  SubstitutionMap smap;

  size_t noperands = std::min(operands.size(), ninputs());
  for (size_t n = 0; n < noperands; n++)
    smap.insert(input(n)->origin(), operands[n]);

  return copy(region, smap);
}

Node *
producer(const jlm::rvsdg::Output * output) noexcept
{
  if (auto node = TryGetOwnerNode<Node>(*output))
    return node;

  if (auto theta = TryGetRegionParentNode<ThetaNode>(*output))
  {
    auto loopvar = theta->MapPreLoopVar(*output);
    if (loopvar.post->origin() != output)
    {
      return nullptr;
    }
    return producer(loopvar.input->origin());
  }

  JLM_ASSERT(dynamic_cast<const RegionArgument *>(output));
  auto argument = static_cast<const RegionArgument *>(output);

  if (!argument->input())
    return nullptr;

  return producer(argument->input()->origin());
}

/**
  \page def_use_inspection Inspecting the graph and matching against different operations

  When inspecting the graph for analysis it is necessary to identify
  different nodes/operations and structures. Depending on the direction,
  the two fundamental questions of interest are:

  - what is the origin of a value, what operation is computing it?
  - what are the users of a particular value, what operations depend on it?

  This requires resolving the type of operation a specific \ref rvsdg::Input
  or \ref rvsdg::output belong to. Every \ref rvsdg::output is one of the following:

  - the output of a node representing an operation
  - the entry argument into a region

  Likewise, every \ref rvsdg::Input is one of the following:

  - the input of a node representing an operation
  - the exit result of a region

  Analysis code can determine which of the two is the case using
  \ref rvsdg::output::GetOwner and \ref rvsdg::Input::GetOwner, respectively,
  and then branch deeper based on its results. For convenience, code
  can more directly match against the specific kinds of nodes using
  the following convenience functions:

  - \ref rvsdg::TryGetOwnerNode checks if the owner of an output/input
    is a graph node of the requested kind
  - \ref rvsdg::TryGetRegionParentNode checks if the output/input is
    a region entry argument / exit result, and if the parent node
    of the region is of the requested kind

  Example:
  \code
  if (auto lambda = rvsdg::TryGetOwnerNode<LambdaNode>(def))
  {
    // This is an output of a lambda node -- so this must
    // be a function definition.
  }
  else if (auto gamma = rvsdg::TryGetOwnerNode<GammaNode>(def))
  {
    // This is an output of a gamma node -- so it is potentially
    // dependent on evaluating a condition.
  }
  else if (auto gamma = rvsdg::TryGetRegionParentNode<GammaNode>(def))
  {
    // This is an entry argument to a region inside a gamma node.
  }
  \endcode

  Similarly, the following variants of the accessor functions
  assert that the nodes are of requested type and will throw
  an exception otherwise:

  - \ref rvsdg::AssertGetOwnerNode asserts that the owner of an
    output/input is a graph node of the requested kind and
    returns it.
  - \ref rvsdg::AssertGetRegionParentNode asserts that the
    output/input is a region entry argument / exit result,
    and that the parent node of the region is of the requested
    kind

  These are mostly suitable for unit tests rather, or for the
  rare circumstances that the type of node can be assumed to
  be known statically.
*/
}

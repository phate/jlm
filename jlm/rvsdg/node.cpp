/*
 * Copyright 2010 2011 2012 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
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

void
Input::CheckTypes(
    const Region & region,
    const Output & origin,
    const std::shared_ptr<const rvsdg::Type> & type)
{
  if (&region != origin.region())
    throw util::Error("Invalid operand region.");

  if (*type != *origin.Type())
    throw util::TypeError(type->debug_string(), origin.Type()->debug_string());
}

Input::Input(rvsdg::Node & owner, rvsdg::Output & origin, std::shared_ptr<const rvsdg::Type> type)
    : index_(0),
      origin_(&origin),
      Owner_(&owner),
      Type_(std::move(type))
{
  CheckTypes(*owner.region(), origin, Type_);
  origin.add_user(this);
}

Input::Input(rvsdg::Region & owner, rvsdg::Output & origin, std::shared_ptr<const rvsdg::Type> type)
    : index_(0),
      origin_(&origin),
      Owner_(&owner),
      Type_(std::move(type))
{
  CheckTypes(owner, origin, Type_);
  origin.add_user(this);
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
    throw util::Error("Invalid operand region.");

  auto old_origin = origin();
  old_origin->remove_user(this);
  this->origin_ = new_origin;
  new_origin->add_user(this);

  if (auto node = TryGetOwnerNode<Node>(*this))
    node->recompute_depth();

  on_input_change(this, old_origin, new_origin);
}

[[nodiscard]] rvsdg::Region *
Input::region() const noexcept
{
  if (auto node = std::get_if<Node *>(&Owner_))
  {
    return (*node)->region();
  }
  else if (auto region = std::get_if<Region *>(&Owner_))
  {
    return *region;
  }
  else
  {
    JLM_UNREACHABLE("Unhandled owner case.");
  }
}

static Input *
ComputeNextInput(const Input * input)
{
  if (input == nullptr)
    return nullptr;

  const auto index = input->index();
  auto owner = input->GetOwner();

  if (auto node = std::get_if<Node *>(&owner))
  {
    return index + 1 < (*node)->ninputs() ? (*node)->input(index + 1) : nullptr;
  }

  if (auto region = std::get_if<Region *>(&owner))
  {
    return index + 1 < (*region)->nresults() ? (*region)->result(index + 1) : nullptr;
  }

  JLM_UNREACHABLE("Unhandled owner case.");
}

Input *
Input::Iterator::ComputeNext() const
{
  return ComputeNextInput(Input_);
}

Input *
Input::ConstIterator::ComputeNext() const
{
  return ComputeNextInput(Input_);
}

Output::~Output() noexcept
{
  JLM_ASSERT(nusers() == 0);
}

Output::Output(Node & owner, std::shared_ptr<const rvsdg::Type> type)
    : index_(0),
      Owner_(&owner),
      Type_(std::move(type))
{}

Output::Output(rvsdg::Region * owner, std::shared_ptr<const rvsdg::Type> type)
    : index_(0),
      Owner_(owner),
      Type_(std::move(type))
{}

[[nodiscard]] rvsdg::Region *
Output::region() const noexcept
{
  if (auto node = std::get_if<Node *>(&Owner_))
  {
    return (*node)->region();
  }
  else if (auto region = std::get_if<Region *>(&Owner_))
  {
    return *region;
  }
  else
  {
    JLM_UNREACHABLE("Unhandled owner case.");
  }
}

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
    if (node->IsDead())
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
    if (node->IsDead())
    {
      bool wasRemoved = region()->RemoveBottomNode(*node);
      JLM_ASSERT(wasRemoved);
    }
  }
  users_.insert(user);
}

static Output *
ComputeNextOutput(const Output * output)
{
  if (output == nullptr)
    return nullptr;

  const auto index = output->index();
  const auto owner = output->GetOwner();

  if (const auto node = std::get_if<Node *>(&owner))
  {
    return index + 1 < (*node)->noutputs() ? (*node)->output(index + 1) : nullptr;
  }

  if (const auto region = std::get_if<Region *>(&owner))
  {
    return index + 1 < (*region)->narguments() ? (*region)->argument(index + 1) : nullptr;
  }

  JLM_UNREACHABLE("Unhandled owner case.");
}

Output *
Output::Iterator::ComputeNext() const
{
  return ComputeNextOutput(Output_);
}

Output *
Output::ConstIterator::ComputeNext() const
{
  return ComputeNextOutput(Output_);
}

node_input::node_input(
    jlm::rvsdg::Output * origin,
    Node * node,
    std::shared_ptr<const rvsdg::Type> type)
    : jlm::rvsdg::Input(*node, *origin, std::move(type))
{}

/* node_output class */

node_output::node_output(Node * node, std::shared_ptr<const rvsdg::Type> type)
    : Output(*node, std::move(type)),
      node_(node)
{}

Node::Node(Region * region)
    : Id_(region->GenerateNodeId()),
      depth_(0),
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

Graph *
Node::graph() const noexcept
{
  return region_->graph();
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
    for (auto & user : output(n)->Users())
    {
      if (auto node = TryGetOwnerNode<Node>(user))
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

const Output &
TraceOutputIntraProcedurally(const Output & output)
{
  // Handle gamma node outputs
  if (const auto gammaNode = TryGetOwnerNode<GammaNode>(output))
  {
    const auto exitVar = gammaNode->MapOutputExitVar(output);
    if (const auto origin = GetGammaInvariantOrigin(*gammaNode, exitVar))
    {
      return TraceOutputIntraProcedurally(*origin.value());
    }

    return output;
  }

  // Handle gamma node arguments
  if (const auto gammaNode = TryGetRegionParentNode<GammaNode>(output))
  {
    const auto roleVar = gammaNode->MapBranchArgument(output);
    if (const auto entryVar = std::get_if<GammaNode::EntryVar>(&roleVar))
    {
      return TraceOutputIntraProcedurally(*entryVar->input->origin());
    }

    if (const auto matchVar = std::get_if<GammaNode::MatchVar>(&roleVar))
    {
      return TraceOutputIntraProcedurally(*matchVar->input->origin());
    }

    return output;
  }

  // Handle theta node outputs
  if (const auto thetaNode = TryGetOwnerNode<ThetaNode>(output))
  {
    const auto loopVar = thetaNode->MapOutputLoopVar(output);
    if (ThetaLoopVarIsInvariant(loopVar))
    {
      return TraceOutputIntraProcedurally(*loopVar.input->origin());
    }

    return output;
  }

  // Handle theta node arguments
  if (const auto thetaNode = TryGetRegionParentNode<ThetaNode>(output))
  {
    const auto loopVar = thetaNode->MapPreLoopVar(output);
    if (ThetaLoopVarIsInvariant(loopVar))
    {
      return TraceOutputIntraProcedurally(*loopVar.input->origin());
    }

    return output;
  }

  return output;
}

/**
  \page def_use_inspection Inspecting the graph and matching against different operations

  When inspecting the graph for analysis it is necessary to identify
  different nodes/operations and structures. Depending on the direction,
  the two fundamental questions of interest are:

  - what is the origin of a value, what operation is computing it?
  - what are the users of a particular value, what operations depend on it?

  This requires resolving the type of operation a specific \ref rvsdg::Input
  or \ref rvsdg::Output belong to. Every \ref rvsdg::Output is one of the following:

  - the output of a node representing an operation
  - the entry argument into a region

  Likewise, every \ref rvsdg::Input is one of the following:

  - the input of a node representing an operation
  - the exit result of a region

  Analysis code can determine which of the two is the case using
  \ref rvsdg::Output::GetOwner and \ref rvsdg::Input::GetOwner, respectively,
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

/*
 * Copyright 2010 2011 2012 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/region.hpp>
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
      Owner_(&owner),
      Type_(std::move(type)),
      UsersList_()
{
  CheckTypes(*owner.region(), origin, Type_);
  origin.add_user(this);
}

Input::Input(rvsdg::Region & owner, rvsdg::Output & origin, std::shared_ptr<const rvsdg::Type> type)
    : index_(0),
      Owner_(&owner),
      Type_(std::move(type)),
      UsersList_()
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
  new_origin->add_user(this);

  region()->notifyInputChange(this, old_origin, new_origin);
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
  JLM_ASSERT(NumUsers_ == 0);
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
  JLM_ASSERT(user->origin_ == this);
  user->origin_ = nullptr;

  Users_.erase(user);
  NumUsers_ -= 1;

  if (auto node = TryGetOwnerNode<Node>(*this))
  {
    node->numSuccessors_ -= 1;
    if (node->IsDead())
    {
      region()->onBottomNodeAdded(*node);
    }
  }
}

void
Output::add_user(jlm::rvsdg::Input * user)
{
  JLM_ASSERT(user->origin_ == nullptr);
  user->origin_ = this;

  if (auto node = TryGetOwnerNode<Node>(*this))
  {
    if (node->IsDead())
    {
      region()->onBottomNodeRemoved(*node);
    }
    node->numSuccessors_ += 1;
  }

  Users_.push_back(user);
  NumUsers_ += 1;
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

NodeInput::NodeInput(
    jlm::rvsdg::Output * origin,
    Node * node,
    std::shared_ptr<const rvsdg::Type> type)
    : jlm::rvsdg::Input(*node, *origin, std::move(type))
{}

NodeOutput::NodeOutput(Node * node, std::shared_ptr<const rvsdg::Type> type)
    : Output(*node, std::move(type))
{}

Node::Node(Region * region)
    : Id_(region->generateNodeId()),
      region_(region)
{
  region->onBottomNodeAdded(*this);
  region->onTopNodeAdded(*this);
  region->onNodeAdded(*this);
}

Node::~Node()
{
  // Nodes should always be dead before they are removed
  JLM_ASSERT(IsDead());
  outputs_.clear();
  region()->onBottomNodeRemoved(*this);

  if (ninputs() == 0)
  {
    region()->onTopNodeRemoved(*this);
  }
  inputs_.clear();

  region()->onNodeRemoved(*this);
}

Graph *
Node::graph() const noexcept
{
  return region_->graph();
}

NodeInput *
Node::addInput(std::unique_ptr<NodeInput> input, bool notifyRegion)
{
  // If we used to be a top node, we no longer are
  if (ninputs() == 0)
  {
    region()->onTopNodeRemoved(*this);
  }

  input->index_ = ninputs();
  inputs_.push_back(std::move(input));
  const auto inputPtr = inputs_.back().get();
  if (notifyRegion)
    region()->notifyInputCreate(inputPtr);

  return inputPtr;
}

void
Node::removeInput(size_t index, bool notifyRegion)
{
  JLM_ASSERT(index < ninputs());

  if (notifyRegion)
    region()->notifyInputDestroy(input(index));

  // remove input
  for (size_t n = index; n < ninputs() - 1; n++)
  {
    inputs_[n] = std::move(inputs_[n + 1]);
    inputs_[n]->index_ = n;
  }
  inputs_.pop_back();

  // If we no longer have any inputs we are now a top node
  if (ninputs() == 0)
  {
    region()->onTopNodeAdded(*this);
  }
}

void
Node::removeOutput(size_t index)
{
  JLM_ASSERT(index < noutputs());
  JLM_ASSERT(outputs_[index]->IsDead());

  for (size_t n = index; n < noutputs() - 1; n++)
  {
    outputs_[n] = std::move(outputs_[n + 1]);
    outputs_[n]->index_ = n;
  }
  outputs_.pop_back();
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

const Output &
traceOutputIntraProcedurally(const Output & output)
{
  // Handle gamma node outputs
  if (const auto gammaNode = TryGetOwnerNode<GammaNode>(output))
  {
    const auto exitVar = gammaNode->MapOutputExitVar(output);
    if (const auto origin = GetGammaInvariantOrigin(*gammaNode, exitVar))
    {
      return traceOutputIntraProcedurally(*origin.value());
    }

    return output;
  }

  // Handle gamma node arguments
  if (const auto gammaNode = TryGetRegionParentNode<GammaNode>(output))
  {
    const auto roleVar = gammaNode->MapBranchArgument(output);
    if (const auto entryVar = std::get_if<GammaNode::EntryVar>(&roleVar))
    {
      return traceOutputIntraProcedurally(*entryVar->input->origin());
    }

    if (const auto matchVar = std::get_if<GammaNode::MatchVar>(&roleVar))
    {
      return traceOutputIntraProcedurally(*matchVar->input->origin());
    }

    return output;
  }

  // Handle theta node outputs
  if (const auto thetaNode = TryGetOwnerNode<ThetaNode>(output))
  {
    const auto loopVar = thetaNode->MapOutputLoopVar(output);
    if (ThetaLoopVarIsInvariant(loopVar))
    {
      return traceOutputIntraProcedurally(*loopVar.input->origin());
    }

    return output;
  }

  // Handle theta node arguments
  if (const auto thetaNode = TryGetRegionParentNode<ThetaNode>(output))
  {
    const auto loopVar = thetaNode->MapPreLoopVar(output);
    if (ThetaLoopVarIsInvariant(loopVar))
    {
      return traceOutputIntraProcedurally(*loopVar.input->origin());
    }

    return output;
  }

  return output;
}

const Output &
traceOutput(const Output & startingOutput)
{
  const auto & output = traceOutputIntraProcedurally(startingOutput);

  // Handle lambda context variables
  if (const auto lambda = rvsdg::TryGetRegionParentNode<rvsdg::LambdaNode>(output))
  {
    // If the argument is a contex variable, continue normalizing
    if (const auto ctxVar = lambda->MapBinderContextVar(output))
      return traceOutput(*ctxVar->input->origin());

    return output;
  }

  // Handle delta context variables
  if (const auto delta = rvsdg::TryGetRegionParentNode<rvsdg::DeltaNode>(output))
  {
    // If the argument is a contex variable, continue normalizing
    const auto ctxVar = delta->MapBinderContextVar(output);
    return traceOutput(*ctxVar.input->origin());
  }

  // Handle phi outputs
  if (const auto phiNode = rvsdg::TryGetOwnerNode<rvsdg::PhiNode>(output))
  {
    const auto fixVar = phiNode->MapOutputFixVar(output);
    return traceOutput(*fixVar.result->origin());
  }

  // Handle phi region arguments
  if (const auto phiNode = rvsdg::TryGetRegionParentNode<rvsdg::PhiNode>(output))
  {
    const auto argument = phiNode->MapArgument(output);
    if (const auto ctxVar = std::get_if<rvsdg::PhiNode::ContextVar>(&argument))
    {
      // Follow the context variable to outside the phi
      return traceOutput(*ctxVar->input->origin());
    }
    if (const auto fixVar = std::get_if<rvsdg::PhiNode::FixVar>(&argument))
    {
      // Follow to the recursion variable's definition
      return traceOutput(*fixVar->result->origin());
    }

    throw std::logic_error("Unknown phi argument type");
  }

  return output;
}

Output &
RouteToRegion(Output & output, Region & region)
{
  if (&region == output.region())
    return output;

  if (&region == &region.graph()->GetRootRegion())
  {
    // We reached the root region and have not found the outputs' region yet.
    // This means that the output comes from a region in the region tree that
    // is not an ancestor of "region".
    throw std::logic_error("Output is not in an ancestor of region.");
  }

  auto & origin = RouteToRegion(output, *region.node()->region());

  const auto newOrigin = MatchTypeOrFail(
      *region.node(),
      [&origin, &region](GammaNode & gammaNode)
      {
        auto [input, branchArgument] = gammaNode.AddEntryVar(&origin);
        return branchArgument[region.index()];
      },
      [&origin](ThetaNode & thetaNode)
      {
        return thetaNode.AddLoopVar(&origin).pre;
      },
      [&origin](LambdaNode & lambdaNode)
      {
        return lambdaNode.AddContextVar(origin).inner;
      },
      [&origin](PhiNode & phiNode)
      {
        return phiNode.AddContextVar(origin).inner;
      },
      [&origin](DeltaNode & deltaNode)
      {
        return deltaNode.AddContextVar(origin).inner;
      });

  return *newOrigin;
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

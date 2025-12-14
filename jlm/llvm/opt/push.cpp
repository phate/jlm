/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/push.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

#include <deque>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/theta.hpp>

namespace jlm::llvm
{

class NodeHoisting::Statistics final : public util::Statistics
{
public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Statistics::Id::PushNodes, sourceFile)
  {}

  void
  start(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgInputsBefore, jlm::rvsdg::ninputs(&graph.GetRootRegion()));
    AddTimer(Label::Timer).start();
  }

  void
  end(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgInputsAfter, jlm::rvsdg::ninputs(&graph.GetRootRegion()));
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

class NodeHoisting::Context final
{
public:
  explicit Context(rvsdg::LambdaNode & lambdaNode)
      : LambdaSubregion_(lambdaNode.subregion())
  {}

  rvsdg::Region &
  getLambdaSubregion() const noexcept
  {
    return *LambdaSubregion_;
  }

  void
  addRegionDepth(const rvsdg::Region & region, const size_t depth) noexcept
  {
    JLM_ASSERT(RegionDepth_.find(&region) == RegionDepth_.end());
    RegionDepth_[&region] = depth;
  }

  size_t
  getRegionDeph(const rvsdg::Region & region) const noexcept
  {
    JLM_ASSERT(RegionDepth_.find(&region) != RegionDepth_.end());
    return RegionDepth_.at(&region);
  }

  void
  addTargetRegion(const rvsdg::Node & node, rvsdg::Region & region) noexcept
  {
    JLM_ASSERT(TargetRegion_.find(&node) == TargetRegion_.end());
    TargetRegion_[&node] = &region;
  }

  rvsdg::Region &
  getTargetRegion(const rvsdg::Node & node) const noexcept
  {
    JLM_ASSERT(TargetRegion_.find(&node) != TargetRegion_.end());
    return *TargetRegion_.at(&node);
  }

  static std::unique_ptr<Context>
  create(rvsdg::LambdaNode & lambdaNode)
  {
    return std::make_unique<Context>(lambdaNode);
  }

private:
  rvsdg::Region * LambdaSubregion_;
  std::unordered_map<const rvsdg::Region *, size_t> RegionDepth_{};
  std::unordered_map<const rvsdg::Node *, rvsdg::Region *> TargetRegion_{};
};

#if 0
class Worklist
{
public:
  inline void
  push_back(rvsdg::Node * node) noexcept
  {
    if (set_.find(node) != set_.end())
      return;

    queue_.push_back(node);
    set_.insert(node);
  }

  rvsdg::Node *
  pop_front() noexcept
  {
    JLM_ASSERT(!empty());
    auto node = queue_.front();
    queue_.pop_front();
    set_.erase(node);
    return node;
  }

  inline bool
  empty() const noexcept
  {
    JLM_ASSERT(queue_.size() == set_.size());
    return queue_.empty();
  }

private:
  std::deque<rvsdg::Node *> queue_;
  std::unordered_set<rvsdg::Node *> set_;
};

static bool
has_side_effects(const rvsdg::Node * node)
{
  for (auto & input : node->Inputs())
  {
    if (input.Type()->Kind() == rvsdg::TypeKind::State)
      return true;
  }

  for (auto & output : node->Outputs())
  {
    if (output.Type()->Kind() == rvsdg::TypeKind::State)
      return true;
  }

  return false;
}

[[nodiscard]] static bool
hasPredecessors(const rvsdg::Node & node)
{
  for (auto & input : node.Inputs())
  {
    if (rvsdg::TryGetOwnerNode<rvsdg::Node>(*input.origin()))
      return true;
  }

  return false;
}

static std::vector<rvsdg::RegionArgument *>
copy_from_gamma(rvsdg::Node * node, size_t r)
{
  JLM_ASSERT(dynamic_cast<rvsdg::GammaNode *>(node->region()->node()));
  JLM_ASSERT(!hasPredecessors(*node));

  auto target = node->region()->node()->region();
  auto gamma = static_cast<rvsdg::GammaNode *>(node->region()->node());

  std::vector<jlm::rvsdg::Output *> operands;
  for (size_t n = 0; n < node->ninputs(); n++)
  {
    JLM_ASSERT(dynamic_cast<const rvsdg::RegionArgument *>(node->input(n)->origin()));
    auto argument = static_cast<const rvsdg::RegionArgument *>(node->input(n)->origin());
    operands.push_back(argument->input()->origin());
  }

  std::vector<rvsdg::RegionArgument *> arguments;
  auto copy = node->copy(target, operands);
  for (size_t n = 0; n < copy->noutputs(); n++)
  {
    auto ev = gamma->AddEntryVar(copy->output(n));
    node->output(n)->divert_users(ev.branchArgument[r]);
    arguments.push_back(util::assertedCast<rvsdg::RegionArgument>(ev.branchArgument[r]));
  }

  return arguments;
}

static std::vector<rvsdg::Output *>
copy_from_theta(rvsdg::Node * node)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::ThetaNode *>(node->region()->node()));
  JLM_ASSERT(!hasPredecessors(*node));

  auto target = node->region()->node()->region();
  auto theta = static_cast<rvsdg::ThetaNode *>(node->region()->node());

  std::vector<jlm::rvsdg::Output *> operands;
  for (size_t n = 0; n < node->ninputs(); n++)
  {
    JLM_ASSERT(dynamic_cast<const rvsdg::RegionArgument *>(node->input(n)->origin()));
    auto argument = static_cast<const rvsdg::RegionArgument *>(node->input(n)->origin());
    operands.push_back(argument->input()->origin());
  }

  std::vector<rvsdg::Output *> arguments;
  auto copy = node->copy(target, operands);
  for (size_t n = 0; n < copy->noutputs(); n++)
  {
    auto lv = theta->AddLoopVar(copy->output(n));
    node->output(n)->divert_users(lv.pre);
    arguments.push_back(lv.pre);
  }

  return arguments;
}

static bool
is_gamma_top_pushable(const rvsdg::Node * node)
{
  return !has_side_effects(node);
}

void
push(rvsdg::GammaNode * gamma)
{
  for (size_t r = 0; r < gamma->nsubregions(); r++)
  {
    auto region = gamma->subregion(r);

    // push out all nullary nodes
    for (auto & node : region->TopNodes())
    {
      if (!has_side_effects(&node))
        copy_from_gamma(&node, r);
    }

    // initialize worklist
    Worklist wl;
    for (size_t n = 0; n < region->narguments(); n++)
    {
      auto argument = region->argument(n);
      for (const auto & user : argument->Users())
      {
        auto tmp = rvsdg::TryGetOwnerNode<rvsdg::Node>(user);
        if (tmp && !hasPredecessors(*tmp))
          wl.push_back(tmp);
      }
    }

    // process worklist
    while (!wl.empty())
    {
      auto node = wl.pop_front();

      if (!is_gamma_top_pushable(node))
        continue;

      auto arguments = copy_from_gamma(node, r);

      // add consumers to worklist
      for (const auto & argument : arguments)
      {
        for (const auto & user : argument->Users())
        {
          auto tmp = rvsdg::TryGetOwnerNode<rvsdg::Node>(user);
          if (tmp && !hasPredecessors(*tmp))
            wl.push_back(tmp);
        }
      }
    }

    region->prune(false);
  }
}

static bool
is_theta_invariant(const rvsdg::Node * node, const std::unordered_set<rvsdg::Output *> & invariants)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::ThetaNode *>(node->region()->node()));
  JLM_ASSERT(!hasPredecessors(*node));

  for (size_t n = 0; n < node->ninputs(); n++)
  {
    JLM_ASSERT(dynamic_cast<const rvsdg::RegionArgument *>(node->input(n)->origin()));
    auto argument = static_cast<rvsdg::RegionArgument *>(node->input(n)->origin());
    if (invariants.find(argument) == invariants.end())
      return false;
  }

  return true;
}

void
push_top(rvsdg::ThetaNode * theta)
{
  auto subregion = theta->subregion();

  // push out all nullary nodes
  for (auto & node : subregion->TopNodes())
  {
    if (!has_side_effects(&node))
      copy_from_theta(&node);
  }

  /* collect loop invariant arguments */
  std::unordered_set<rvsdg::Output *> invariants;
  for (const auto & lv : theta->GetLoopVars())
  {
    if (lv.post->origin() == lv.pre)
      invariants.insert(lv.pre);
  }

  // initialize worklist
  Worklist wl;
  for (const auto & lv : theta->GetLoopVars())
  {
    auto argument = lv.pre;
    for (const auto & user : argument->Users())
    {
      auto tmp = rvsdg::TryGetOwnerNode<rvsdg::Node>(user);
      if (tmp && !hasPredecessors(*tmp) && is_theta_invariant(tmp, invariants))
        wl.push_back(tmp);
    }
  }

  // process worklist
  while (!wl.empty())
  {
    auto node = wl.pop_front();

    /* we cannot push out nodes with side-effects */
    if (has_side_effects(node))
      continue;

    auto arguments = copy_from_theta(node);
    invariants.insert(arguments.begin(), arguments.end());

    // add consumers to worklist
    for (const auto & argument : arguments)
    {
      for (const auto & user : argument->Users())
      {
        auto tmp = rvsdg::TryGetOwnerNode<rvsdg::Node>(user);
        if (tmp && !hasPredecessors(*tmp) && is_theta_invariant(tmp, invariants))
          wl.push_back(tmp);
      }
    }
  }
}

static bool
is_invariant(const rvsdg::RegionArgument * argument)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::ThetaNode *>(argument->region()->node()));
  return argument->region()->result(argument->index() + 1)->origin() == argument;
}

static bool
is_movable_store(rvsdg::Node * node)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::ThetaNode *>(node->region()->node()));
  JLM_ASSERT(jlm::rvsdg::is<StoreNonVolatileOperation>(node));

  auto address = dynamic_cast<rvsdg::RegionArgument *>(node->input(0)->origin());
  if (!address || !is_invariant(address) || address->nusers() != 2)
    return false;

  for (size_t n = 2; n < node->ninputs(); n++)
  {
    auto argument = dynamic_cast<rvsdg::RegionArgument *>(node->input(n)->origin());
    if (!argument || argument->nusers() > 1)
      return false;
  }

  for (size_t n = 0; n < node->noutputs(); n++)
  {
    auto output = node->output(n);
    if (output->nusers() != 1)
      return false;

    if (!dynamic_cast<rvsdg::RegionResult *>(&output->SingleUser()))
      return false;
  }

  return true;
}

static void
pushout_store(rvsdg::SimpleNode * storenode)
{
  JLM_ASSERT(dynamic_cast<const rvsdg::ThetaNode *>(storenode->region()->node()));
  JLM_ASSERT(jlm::rvsdg::is<StoreNonVolatileOperation>(storenode) && is_movable_store(storenode));
  auto theta = static_cast<rvsdg::ThetaNode *>(storenode->region()->node());
  auto storeop = static_cast<const StoreNonVolatileOperation *>(&storenode->GetOperation());
  auto oaddress = static_cast<rvsdg::RegionArgument *>(storenode->input(0)->origin());
  auto ovalue = storenode->input(1)->origin();

  /* insert new value for store */
  auto nvalue = theta->AddLoopVar(UndefValueOperation::Create(*theta->region(), ovalue->Type()));
  nvalue.post->divert_to(ovalue);

  /* collect store operands */
  std::vector<jlm::rvsdg::Output *> states;
  auto address = oaddress->input()->origin();
  for (size_t n = 0; n < storenode->noutputs(); n++)
  {
    JLM_ASSERT(storenode->output(n)->nusers() == 1);
    auto result = static_cast<rvsdg::RegionResult *>(&storenode->output(n)->SingleUser());
    result->divert_to(storenode->input(n + 2)->origin());
    states.push_back(result->output());
  }

  /* create new store and redirect theta output users */
  auto nstates =
      StoreNonVolatileOperation::Create(address, nvalue.output, states, storeop->GetAlignment());
  for (size_t n = 0; n < states.size(); n++)
  {
    std::unordered_set<jlm::rvsdg::Input *> users;
    for (auto & user : states[n]->Users())
    {
      if (rvsdg::TryGetOwnerNode<rvsdg::Node>(user)
          != rvsdg::TryGetOwnerNode<rvsdg::Node>(*nstates[0]))
        users.insert(&user);
    }

    for (const auto & user : users)
      user->divert_to(nstates[n]);
  }

  remove(storenode);
}

void
push_bottom(rvsdg::ThetaNode * theta)
{
  for (const auto & lv : theta->GetLoopVars())
  {
    const auto storeNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*lv.post->origin());
    if (jlm::rvsdg::is<StoreNonVolatileOperation>(storeNode) && is_movable_store(storeNode))
    {
      pushout_store(storeNode);
      break;
    }
  }
}

void
push(rvsdg::ThetaNode * theta)
{
  bool done = false;
  while (!done)
  {
    auto nnodes = theta->subregion()->numNodes();
    push_top(theta);
    push_bottom(theta);
    if (nnodes == theta->subregion()->numNodes())
      done = true;
  }

  theta->subregion()->prune(false);
}

static void
push(rvsdg::Region * region)
{
  for (auto node : rvsdg::TopDownTraverser(region))
  {
    if (auto strnode = dynamic_cast<const rvsdg::StructuralNode *>(node))
    {
      for (size_t n = 0; n < strnode->nsubregions(); n++)
        push(strnode->subregion(n));
    }

    if (auto gamma = dynamic_cast<rvsdg::GammaNode *>(node))
      push(gamma);

    if (auto theta = dynamic_cast<rvsdg::ThetaNode *>(node))
      push(theta);
  }
}

static void
push(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
{
  auto statistics = NodeHoisting::Statistics::Create(rvsdgModule.SourceFilePath().value());

  statistics->start(rvsdgModule.Rvsdg());
  push(&rvsdgModule.Rvsdg().GetRootRegion());
  statistics->end(rvsdgModule.Rvsdg());

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}
#endif

NodeHoisting::~NodeHoisting() noexcept = default;

NodeHoisting::NodeHoisting()
    : Transformation("NodeHoisting")
{}

size_t
NodeHoisting::computeRegionDepth(const rvsdg::Region & region) const
{
  if (dynamic_cast<const rvsdg::LambdaNode *>(region.node()))
  {
    return 0;
  }

  const auto parentRegion = region.node()->region();
  return Context_->getRegionDeph(*parentRegion) + 1;
}

bool
NodeHoisting::isInvariantMemoryStateLoopVar(const rvsdg::ThetaNode::LoopVar & loopVar)
{
  if (!is<MemoryStateType>(loopVar.output->Type()))
    return false;

  if (loopVar.pre->nusers() != 1)
    return false;

  const auto userNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*loopVar.pre->Users().begin());
  const auto originNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*loopVar.post->origin());

  if (userNode != originNode)
    return false;

  return true;
}

rvsdg::Region &
NodeHoisting::computeTargetRegion(const rvsdg::Output & output) const
{
  if (is<IOStateType>(output.Type()))
  {
    // We never hoist a node with an IOState.
    return *output.region();
  }

  // Handle lambda region arguments
  if (rvsdg::TryGetRegionParentNode<rvsdg::LambdaNode>(output))
  {
    return *output.region();
  }

  // Handle gamma region arguments
  if (const auto gammaNode = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(output))
  {
    const auto roleVar = gammaNode->MapBranchArgument(output);
    if (const auto entryVar = std::get_if<rvsdg::GammaNode::EntryVar>(&roleVar))
    {
      return computeTargetRegion(*entryVar->input->origin());
    }

    return *output.region();
  }

  // Handle theta region arguments
  if (const auto thetaNode = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    const auto loopVar = thetaNode->MapPreLoopVar(output);
    if (rvsdg::ThetaLoopVarIsInvariant(loopVar))
    {
      return computeTargetRegion(*loopVar.input->origin());
    }

    if (isInvariantMemoryStateLoopVar(loopVar))
    {
      return computeTargetRegion(*loopVar.input->origin());
    }

    return *output.region();
  }

  // Handle gamma outputs
  if (const auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(output))
  {
    return Context_->getTargetRegion(*gammaNode);
  }

  // Handle theta outputs
  if (const auto thetaNode = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(output))
  {
    return Context_->getTargetRegion(*thetaNode);
  }

  // Handle simple node outputs
  if (const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
  {
    return Context_->getTargetRegion(*node);
  }

  throw std::logic_error("Unhandled output type!");
}

rvsdg::Region &
NodeHoisting::computeTargetRegion(const rvsdg::Node & node) const
{
  if (node.ninputs() == 0)
  {
    // Nodes without inputs can always be hoisted to the lambda region
    return Context_->getLambdaSubregion();
  }

  // Compute target regions for all the inputs of the node
  std::vector<rvsdg::Region *> targetRegions;
  for (auto & input : node.Inputs())
  {
    auto & targetRegion = computeTargetRegion(*input.origin());
    if (&targetRegion == node.region())
    {
      // One of the node's predecessors cannot be hoisted, which means we can also not hoist this
      // node
      return *node.region();
    }

    targetRegions.push_back(&targetRegion);
  }

  // Compute the lowermost target region in the region tree
  return **std::max_element(
      targetRegions.begin(),
      targetRegions.end(),
      [&](const rvsdg::Region * region1, const rvsdg::Region * region2)
      {
        return Context_->getRegionDeph(*region1) < Context_->getRegionDeph(*region2);
      });
}

void
NodeHoisting::markNodes(const rvsdg::Region & region)
{
  const auto regionDepth = computeRegionDepth(region);
  Context_->addRegionDepth(region, regionDepth);

  for (const auto node : rvsdg::TopDownConstTraverser(&region))
  {
    rvsdg::MatchTypeWithDefault(
        *node,
        [&](const rvsdg::StructuralNode & structuralNode)
        {
          // FIXME: We currently do not allow structural nodes (gamma and theta nodes) to be hoisted
          Context_->addTargetRegion(structuralNode, *structuralNode.region());

          // Handle innermost regions
          for (auto & subregion : structuralNode.Subregions())
          {
            markNodes(subregion);
          }
        },
        [&](const rvsdg::SimpleNode & simpleNode)
        {
          rvsdg::Region & targetRegion = computeTargetRegion(simpleNode);
          Context_->addTargetRegion(*node, targetRegion);
        },
        []()
        {
          throw std::logic_error("Unhandled node type!");
        });
  }
}

rvsdg::Output &
NodeHoisting::getOperandFromTargetRegion(rvsdg::Output & output, rvsdg::Region & targetRegion)
{
  if (output.region() == &targetRegion)
    return output;

  // Handle gamma subregion arguments
  if (const auto gammaNode = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(output))
  {
    const auto roleVar = gammaNode->MapBranchArgument(output);
    if (const auto entryVar = std::get_if<rvsdg::GammaNode::EntryVar>(&roleVar))
    {
      return getOperandFromTargetRegion(*entryVar->input->origin(), targetRegion);
    }
  }

  // Handle theta subregion arguments
  if (const auto thetaNode = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    const auto loopVar = thetaNode->MapPreLoopVar(output);
    JLM_ASSERT(rvsdg::ThetaLoopVarIsInvariant(loopVar) || isInvariantMemoryStateLoopVar(loopVar));
    return getOperandFromTargetRegion(*loopVar.input->origin(), targetRegion);
  }

  throw std::logic_error("Unhandled output type!");
}

std::vector<rvsdg::Output *>
NodeHoisting::getOperandsFromTargetRegion(rvsdg::Node & node, rvsdg::Region & targetRegion)
{
  std::vector<rvsdg::Output *> operands;
  for (auto & input : node.Inputs())
  {
    auto & operand = getOperandFromTargetRegion(*input.origin(), targetRegion);
    operands.push_back(&operand);
  }

  return operands;
}

void
NodeHoisting::copyNodeToTargetRegion(rvsdg::Node & node) const
{
  auto & targetRegion = Context_->getTargetRegion(node);
  JLM_ASSERT(&targetRegion != node.region());

  const auto operands = getOperandsFromTargetRegion(node, targetRegion);
  const auto copiedNode = node.copy(&targetRegion, operands);

  // FIXME: I really would like to have a zip function here, but C++ does not really seem to have
  // anything better to offer
  auto itOrg = std::begin(node.Outputs());
  const auto endOrg = std::end(node.Outputs());
  auto itCpy = std::begin(copiedNode->Outputs());
  const auto endCpy = std::end(copiedNode->Outputs());
  JLM_ASSERT(std::distance(itOrg, endOrg) == std::distance(itCpy, endCpy));

  for (; itOrg != endOrg; ++itOrg, ++itCpy)
  {
    auto & outputOrg = *itOrg;
    auto & outputCpy = *itCpy;
    auto & newOutputOrg = rvsdg::RouteToRegion(outputCpy, *node.region());
    outputOrg.divert_users(&newOutputOrg);
  }
}

void
NodeHoisting::hoistNodes(rvsdg::Region & region)
{
  // FIXME: We a routing unnecessary values through gamma and theta nodes. We should cluster
  // subgraphs that need to be hoisted to avoid unnecessary routing.
  for (const auto node : rvsdg::TopDownTraverser(&region))
  {
    auto & targetRegion = Context_->getTargetRegion(*node);
    if (&targetRegion != node->region())
    {
      copyNodeToTargetRegion(*node);
    }

    // Handle innermost regions
    if (const auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        hoistNodes(subregion);
      }
    }
  }

  region.prune(false);
}

void
NodeHoisting::hoistNodesInLambda(rvsdg::LambdaNode & lambdaNode)
{
  Context_ = Context::create(lambdaNode);

  markNodes(*lambdaNode.subregion());
  hoistNodes(*lambdaNode.subregion());

  Context_.reset();
}

void
NodeHoisting::hoistNodesInRootRegion(rvsdg::Region & region)
{
  for (auto & node : rvsdg::TopDownTraverser(&region))
  {
    rvsdg::MatchTypeWithDefault(
        *node,
        [&](rvsdg::LambdaNode & lambdaNode)
        {
          hoistNodesInLambda(lambdaNode);
        },
        [&](rvsdg::PhiNode & phiNode)
        {
          hoistNodesInRootRegion(*phiNode.subregion());
        },
        [](rvsdg::DeltaNode &)
        {
          // Nothing needs to be done
        },
        [](rvsdg::SimpleNode &)
        {
          // Nothing needs to be done
        },
        [&]()
        {
          throw std::logic_error(util::strfmt("Unhandled node type: ", node->DebugString()));
        });
  }
}

void
NodeHoisting::Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFilePath().value());

  statistics->start(rvsdgModule.Rvsdg());
  hoistNodesInRootRegion(rvsdgModule.Rvsdg().GetRootRegion());
  statistics->end(rvsdgModule.Rvsdg());

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}
}

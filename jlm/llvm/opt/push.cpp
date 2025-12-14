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
  // Handle lambda region arguments
  if (rvsdg::TryGetRegionParentNode<rvsdg::LambdaNode>(output))
  {
    return *output.region();
  }

  // Handle gamma region arguments
  if (const auto gammaNode = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(output))
  {
    if (output.Type()->Kind() == rvsdg::TypeKind::State)
    {
      // FIXME: This is a bit too conservative. For example, it avoids that load and store nodes are
      // hoisted out of a gamma node, but we would only like to avoid store nodes being hoisted out.
      // For load nodes, it is legal to hoist them out if they are not preceded by an IOBarrier.
      return *output.region();
    }

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

    auto [constantNode, constantOp] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::ControlConstantOperation>(node);
    if (constantOp)
    {
      // FIXME: We currently need to leave control constants where they are even though there is
      // no inherent constraint in the operation for hoisting them. The reason is that the control
      // type is of kind state, which means that the LLVM back-end filters it out. However, if we
      // would hoist this operation out, it could be that we would need to create select operations
      // for it (which cannot be materialized in LLVM as it is of kind state), which in turn leads
      // to problems with SSA phi node creation as a phi operand might be such a select operation.
      //
      // The fundamental problem is that we model control type with kind state, even though we need
      // to materialize it in the back-end. We should change its kind to value.
      return *node.region();
    }

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

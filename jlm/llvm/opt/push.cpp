/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/push.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

#include <algorithm>
#include <deque>

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
  addTargetRegion(const rvsdg::Node & node, rvsdg::Region & region) noexcept
  {
    JLM_ASSERT(TargetRegion_.find(&node) == TargetRegion_.end());
    TargetRegion_[&node] = &region;
  }

  rvsdg::Region &
  getTargetRegion(const rvsdg::Node & node) const noexcept
  {
    return *TargetRegion_.at(&node);
  }

  static std::unique_ptr<Context>
  create(rvsdg::LambdaNode & lambdaNode)
  {
    return std::make_unique<Context>(lambdaNode);
  }

private:
  rvsdg::Region * LambdaSubregion_;
  std::unordered_map<const rvsdg::Node *, rvsdg::Region *> TargetRegion_{};
};

NodeHoisting::~NodeHoisting() noexcept = default;

NodeHoisting::NodeHoisting()
    : Transformation("NodeHoisting")
{}

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
    if (is<IOStateType>(output.Type()))
    {
      // Do not hoist nodes with IO state edges out of gamma nodes.
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
    return context_->getTargetRegion(*gammaNode);
  }

  // Handle theta outputs
  if (const auto thetaNode = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(output))
  {
    return context_->getTargetRegion(*thetaNode);
  }

  // Handle simple node outputs
  if (const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
  {
    return context_->getTargetRegion(*node);
  }

  throw std::logic_error("Unhandled output type!");
}

static bool
hasOnlyValueInputs(const rvsdg::Node & node)
{
  for (auto & input : node.Inputs())
  {
    if (input.Type()->Kind() != rvsdg::TypeKind::Value)
      return false;
  }

  return true;
}

static rvsdg::Region &
limitTargetRegion(const rvsdg::Node & node, rvsdg::Region & targetRegion)
{
  JLM_ASSERT(node.region() != &targetRegion);

  if (hasOnlyValueInputs(node))
  {
    // Pure nodes can be hoisted out of gamma and theta nodes.
    return targetRegion;
  }

  if (is<LoadNonVolatileOperation>(node.GetOperation()))
  {
    // LoadNonVolatileOperation nodes can also be hoisted out of gamma and theta nodes.
    return targetRegion;
  }

  // For all other nodes, we want to limit the target region to the lowest gamma node.
  auto currentRegion = node.region();
  do
  {
    if (dynamic_cast<rvsdg::GammaNode *>(currentRegion->node()))
    {
      break;
    }

    currentRegion = currentRegion->node()->region();
  } while (currentRegion != &targetRegion);

  return *currentRegion;
}

rvsdg::Region &
NodeHoisting::computeTargetRegion(const rvsdg::Node & node) const
{
  if (node.ninputs() == 0)
  {
    // All nullary operations to date have exactly one output
    JLM_ASSERT(node.noutputs() == 1);

    // Control constants are used to instruct the control flow graph creation,
    // and will be removed in the back-end, so there is no need to hoist them.
    const auto outputType = node.output(0)->Type();
    if (is<rvsdg::ControlType>(outputType))
      return *node.region();

    // Other constants should be moved to the top-level of the function
    return context_->getLambdaSubregion();
  }

  // Compute target regions for all the inputs of the node
  rvsdg::Region * greatestCommonTargetRegion = nullptr;

  for (auto & input : node.Inputs())
  {
    auto & targetRegion = computeTargetRegion(*input.origin(), node.GetOperation());
    if (&targetRegion == node.region())
    {
      // One of the node's predecessors cannot be hoisted, which means we can also not hoist this
      // node
      return *node.region();
    }

    // If we already have a common target region that is lower, keep it
    if (greatestCommonTargetRegion
        && greatestCommonTargetRegion->getDepth() >= targetRegion.getDepth())
      continue;
    greatestCommonTargetRegion = &targetRegion;
  }

  greatestCommonTargetRegion = &limitTargetRegion(node, *greatestCommonTargetRegion);

  // Return the lowest-most common target region in the region tree among all inputs
  JLM_ASSERT(greatestCommonTargetRegion);
  return *greatestCommonTargetRegion;
}

void
NodeHoisting::markNodes(const rvsdg::Region & region)
{
  for (const auto node : rvsdg::TopDownConstTraverser(&region))
  {
    rvsdg::MatchTypeWithDefault(
        *node,
        [&](const rvsdg::StructuralNode & structuralNode)
        {
          // FIXME: We currently do not allow structural nodes (gamma and theta nodes) to be hoisted
          context_->addTargetRegion(structuralNode, *structuralNode.region());

          // Handle innermost regions
          for (auto & subregion : structuralNode.Subregions())
          {
            markNodes(subregion);
          }
        },
        [&](const rvsdg::SimpleNode & simpleNode)
        {
          rvsdg::Region & targetRegion = computeTargetRegion(simpleNode);
          context_->addTargetRegion(*node, targetRegion);
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

static rvsdg::Input *
mapStateOutputToInput(rvsdg::Output & output)
{
  JLM_ASSERT(output.Type()->Kind() == rvsdg::TypeKind::State);

  const auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output);
  JLM_ASSERT(simpleNode);

  return rvsdg::MatchTypeWithDefault(
      simpleNode->GetOperation(),
      [&output](const LoadNonVolatileOperation &)
      {
        return &LoadOperation::MapMemoryStateOutputToInput(output);
      },
      [&output](const StoreNonVolatileOperation &)
      {
        return &StoreOperation::MapMemoryStateOutputToInput(output);
      },
      [](const VariadicArgumentListOperation &) -> rvsdg::Input *
      {
        return nullptr;
      },
      [](const CallEntryMemoryStateMergeOperation &) -> rvsdg::Input *
      {
        return nullptr;
      },
      [](const AllocaOperation &) -> rvsdg::Input *
      {
        return nullptr;
      },
      [&simpleNode]() -> rvsdg::Input *
      {
        throw std::logic_error(
            util::strfmt("Unhandled operation type: ", simpleNode->DebugString()));
      });
}

void
NodeHoisting::copyNodeToTargetRegion(rvsdg::Node & node) const
{
  auto & targetRegion = context_->getTargetRegion(node);
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

    if (outputOrg.Type()->Kind() == rvsdg::TypeKind::State)
    {
      if (auto inputOrg = mapStateOutputToInput(outputOrg))
      {
        outputOrg.divert_users(inputOrg->origin());

        auto inputCpy = mapStateOutputToInput(outputCpy);
        JLM_ASSERT(inputCpy);

        // FIXME: We introduce a slight impression here. If inputCpy->origin() has
        // more than a single user, then all users will all in a sudden be
        // sequentialized after the hoisted node even though they were only
        // sequentialized by the producer of inputCpy->origin() before.
        inputCpy->origin()->divertUsersWhere(
            outputCpy,
            [&inputCpy](const rvsdg::Input & input)
            {
              return &input != inputCpy;
            });
      }
      else
      {
        // If we cannot map the output state to the input state of the node, we fall back value-edge
        // semantic for hoisting.
        auto & newOutputOrg = rvsdg::RouteToRegion(outputCpy, *node.region());
        outputOrg.divert_users(&newOutputOrg);
      }
    }
    else if (outputOrg.Type()->Kind() == rvsdg::TypeKind::Value)
    {
      auto & newOutputOrg = rvsdg::RouteToRegion(outputCpy, *node.region());
      outputOrg.divert_users(&newOutputOrg);
    }
    else
    {
      throw std::logic_error(util::strfmt("Unhandled type kind!"));
    }
  }
}

void
NodeHoisting::hoistNodes(rvsdg::Region & region)
{
  // FIXME: We a routing unnecessary values through gamma and theta nodes. We should cluster
  // subgraphs that need to be hoisted to avoid unnecessary routing.
  for (const auto node : rvsdg::TopDownTraverser(&region))
  {
    auto & targetRegion = context_->getTargetRegion(*node);
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
  context_ = Context::create(lambdaNode);

  markNodes(*lambdaNode.subregion());
  hoistNodes(*lambdaNode.subregion());

  context_.reset();
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

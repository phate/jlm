/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::llvm
{

/** \brief Dead Node Elimination context class
 *
 * This class keeps track of all the nodes and outputs that are alive. In contrast to all other
 * nodes, a simple node is considered alive if already a single of its outputs is alive. For this
 * reason, this class keeps separately track of simple nodes and therefore avoids to store all its
 * outputs (and instead stores the node itself). By marking the entire node as alive, we also avoid
 * that we reiterate through all inputs of this node again in the future. The following example
 * illustrates the issue:
 *
 * o1 ... oN = Node2 i1 ... iN
 * p1 ... pN = Node1 o1 ... oN
 *
 * When we mark o1 as alive, we actually mark the entire Node2 as alive. This means that when we try
 * to mark o2 alive in the future, we can immediately stop marking instead of reiterating through i1
 * ... iN again. Thus, by marking the entire simple node instead of just its outputs, we reduce the
 * runtime for marking Node2 from O(oN x iN) to O(oN + iN).
 */
class DeadNodeElimination::Context final
{
public:
  void
  markAlive(const rvsdg::Output & output)
  {
    Outputs_.insert(&output);
  }

  void
  markAlive(const rvsdg::SimpleNode & simpleNode)
  {
    SimpleNodes_.insert(&simpleNode);
  }

  bool
  isAlive(const rvsdg::Output & output) const noexcept
  {
    return Outputs_.Contains(&output);
  }

  bool
  isAlive(const rvsdg::SimpleNode & simpleNode) const noexcept
  {
    return SimpleNodes_.Contains(&simpleNode);
  }

  static std::unique_ptr<Context>
  create()
  {
    return std::make_unique<Context>();
  }

private:
  util::HashSet<const rvsdg::SimpleNode *> SimpleNodes_{};
  util::HashSet<const rvsdg::Output *> Outputs_{};
};

/** \brief Dead Node Elimination statistics class
 *
 */
class DeadNodeElimination::Statistics final : public util::Statistics
{
  const char * MarkTimerLabel_ = "MarkTime";
  const char * SweepTimerLabel_ = "SweepTime";

public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Id::DeadNodeElimination, sourceFile)
  {}

  void
  startMarkStatistics(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(&graph.GetRootRegion()));
    AddMeasurement(Label::NumRvsdgInputsBefore, rvsdg::ninputs(&graph.GetRootRegion()));
    AddTimer(MarkTimerLabel_).start();
  }

  void
  stopMarkStatistics() noexcept
  {
    GetTimer(MarkTimerLabel_).stop();
  }

  void
  startSweepStatistics() noexcept
  {
    AddTimer(SweepTimerLabel_).start();
  }

  void
  stopSweepStatistics(const rvsdg::Graph & graph) noexcept
  {
    GetTimer(SweepTimerLabel_).stop();
    AddMeasurement(Label::NumRvsdgNodesAfter, rvsdg::nnodes(&graph.GetRootRegion()));
    AddMeasurement(Label::NumRvsdgInputsAfter, rvsdg::ninputs(&graph.GetRootRegion()));
  }

  static std::unique_ptr<Statistics>
  create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

DeadNodeElimination::~DeadNodeElimination() noexcept = default;

DeadNodeElimination::DeadNodeElimination()
    : Transformation("DeadNodeElimination")
{}

void
DeadNodeElimination::run(rvsdg::Region & region)
{
  Context_ = Context::create();

  markRegion(region);
  sweepRegion(region);

  // Discard internal state to free up memory after we are done
  Context_.reset();
}

void
DeadNodeElimination::Run(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  Context_ = Context::create();

  auto & rvsdg = module.Rvsdg();
  auto statistics = Statistics::create(module.SourceFilePath().value());
  statistics->startMarkStatistics(rvsdg);
  markRegion(rvsdg.GetRootRegion());
  statistics->stopMarkStatistics();

  statistics->startSweepStatistics();
  sweepRvsdg(rvsdg);
  statistics->stopSweepStatistics(rvsdg);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Discard internal state to free up memory after we are done
  Context_.reset();
}

static bool
isLoadNonVolatileMemoryStateOutput(const rvsdg::Output & output)
{
  const auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output);
  return simpleNode && is<LoadNonVolatileOperation>(simpleNode)
      && is<MemoryStateType>(output.Type());
}

void
DeadNodeElimination::markRegion(const rvsdg::Region & region)
{
  for (const auto result : region.Results())
  {
    markOutput(*result->origin());
  }
}

void
DeadNodeElimination::markOutput(const rvsdg::Output & output)
{
  auto isAlive = [this](const rvsdg::Output & output)
  {
    if (const auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
    {
      return Context_->isAlive(*simpleNode);
    }

    return Context_->isAlive(output);
  };

  auto markAlive = [this](const rvsdg::Output & output)
  {
    // FIXME: Avoiding to mark load nodes as alive via the memory states might lead to performance
    // issues. If we have multiple load nodes sequentialized after each other, each with several
    // memory states, then we will visit them multiple times. A solution to this problem might be:
    //
    // 1. Separate the memoization of the visited nodes/outputs from the nodes/outputs that are
    // alive, i.e., introduce an alive and visisted set.
    //
    // 2. The additional memoization from above might lead to increased memory consumption, so we
    // maybe would like to start to interleave the mark and sweep phases. Instead of performing the
    // marking on the entire module followed by sweeping, we can do mark/sweep on function-level or
    // even more fine-grained on region level. This would allow us to deallocate the stored
    // memoization after each region sweep, relaxing the memory footprint of the pass.
    if (isLoadNonVolatileMemoryStateOutput(output))
      return;

    if (const auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
    {
      Context_->markAlive(*simpleNode);
    }
    else
    {
      Context_->markAlive(output);
    }
  };

  if (isAlive(output))
  {
    return;
  }

  markAlive(output);

  if (is<rvsdg::GraphImport>(&output))
  {
    return;
  }

  if (const auto gamma = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(output))
  {
    markOutput(*gamma->predicate()->origin());
    for (const auto & result : gamma->MapOutputExitVar(output).branchResult)
    {
      markOutput(*result->origin());
    }
    return;
  }

  if (const auto gamma = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(output))
  {
    const auto origin = std::visit(
        [](const auto & roleVar) -> rvsdg::Output *
        {
          return roleVar.input->origin();
        },
        gamma->MapBranchArgument(output));
    markOutput(*origin);
    return;
  }

  if (const auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(output))
  {
    const auto loopVar = theta->MapOutputLoopVar(output);
    markOutput(*theta->predicate()->origin());
    markOutput(*loopVar.post->origin());
    markOutput(*loopVar.input->origin());
    return;
  }

  if (const auto theta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    const auto loopVar = theta->MapPreLoopVar(output);
    markOutput(*loopVar.output);
    markOutput(*loopVar.input->origin());
    return;
  }

  if (const auto lambda = rvsdg::TryGetOwnerNode<rvsdg::LambdaNode>(output))
  {
    for (const auto result : lambda->GetFunctionResults())
    {
      markOutput(*result->origin());
    }
    return;
  }

  if (const auto lambda = rvsdg::TryGetRegionParentNode<rvsdg::LambdaNode>(output))
  {
    if (const auto ctxVar = lambda->MapBinderContextVar(output))
    {
      // Bound context variable.
      markOutput(*ctxVar->input->origin());
      return;
    }

    // Function argument.
    return;
  }

  if (const auto phi = rvsdg::TryGetOwnerNode<rvsdg::PhiNode>(output))
  {
    markOutput(*phi->MapOutputFixVar(output).result->origin());
    return;
  }

  if (const auto phi = rvsdg::TryGetRegionParentNode<rvsdg::PhiNode>(output))
  {
    const auto var = phi->MapArgument(output);
    if (const auto fixVar = std::get_if<rvsdg::PhiNode::FixVar>(&var))
    {
      // Recursion argument
      markOutput(*fixVar->result->origin());
      return;
    }

    if (const auto ctxVar = std::get_if<rvsdg::PhiNode::ContextVar>(&var))
    {
      // Bound context variable.
      markOutput(*ctxVar->input->origin());
      return;
    }

    throw std::logic_error("Phi argument must be either fixpoint or context variable");
  }

  if (const auto deltaNode = rvsdg::TryGetOwnerNode<rvsdg::DeltaNode>(output))
  {
    const auto result = deltaNode->subregion()->result(0);
    markOutput(*result->origin());
    return;
  }

  if (rvsdg::TryGetRegionParentNode<rvsdg::DeltaNode>(output))
  {
    const auto argument = util::assertedCast<const rvsdg::RegionArgument>(&output);
    markOutput(*argument->input()->origin());
    return;
  }

  if (const auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
  {
    if (isLoadNonVolatileMemoryStateOutput(output))
    {
      markOutput(*LoadOperation::MapMemoryStateOutputToInput(output).origin());
    }
    else
    {
      for (auto & input : simpleNode->Inputs())
      {
        markOutput(*input.origin());
      }
    }
    return;
  }

  throw std::logic_error("We should have never reached this statement.");
}

void
DeadNodeElimination::sweepRvsdg(rvsdg::Graph & rvsdg) const
{
  sweepRegion(rvsdg.GetRootRegion());

  // Remove dead imports
  util::HashSet<size_t> indices;
  for (const auto argument : rvsdg.GetRootRegion().Arguments())
  {
    if (!Context_->isAlive(*argument))
    {
      indices.insert(argument->index());
    }
  }
  [[maybe_unused]] const auto numRemovedArguments = rvsdg.GetRootRegion().RemoveArguments(indices);
  JLM_ASSERT(numRemovedArguments == indices.Size());
}

void
DeadNodeElimination::sweepRegion(rvsdg::Region & region) const
{
  auto isAlive = [this](const rvsdg::Node & node)
  {
    if (const auto simpleNode = dynamic_cast<const rvsdg::SimpleNode *>(&node))
    {
      return Context_->isAlive(*simpleNode);
    }

    for (auto & output : node.Outputs())
    {
      if (Context_->isAlive(output))
      {
        return true;
      }
    }

    return false;
  };

  for (const auto node : rvsdg::BottomUpTraverser(&region))
  {
    if (!isAlive(*node))
    {
      removeNode(*node);
    }
    else if (const auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      sweepStructuralNode(*structuralNode);
    }
  }

  JLM_ASSERT(region.numBottomNodes() == 0);
}

void
DeadNodeElimination::sweepStructuralNode(rvsdg::StructuralNode & node) const
{
  rvsdg::MatchTypeWithDefault(
      node,
      [this](rvsdg::GammaNode & node)
      {
        sweepGamma(node);
      },
      [this](rvsdg::ThetaNode & node)
      {
        sweepTheta(node);
      },
      [this](rvsdg::LambdaNode & node)
      {
        sweepLambda(node);
      },
      [this](rvsdg::PhiNode & node)
      {
        sweepPhi(node);
      },
      [](rvsdg::DeltaNode & node)
      {
        sweepDelta(node);
      },
      [&node]()
      {
        throw std::logic_error(util::strfmt("Unhandled node type: ", node.DebugString()));
      });
}

void
DeadNodeElimination::sweepGamma(rvsdg::GammaNode & gammaNode) const
{
  // Remove dead exit vars.
  std::vector<rvsdg::GammaNode::ExitVar> deadExitVars;
  for (const auto & exitVar : gammaNode.GetExitVars())
  {
    if (!Context_->isAlive(*exitVar.output))
    {
      deadExitVars.push_back(exitVar);
    }
  }
  gammaNode.RemoveExitVars(deadExitVars);

  // Sweep gamma subregions
  for (auto & subregion : gammaNode.Subregions())
  {
    sweepRegion(subregion);
  }

  // Remove dead entry vars.
  std::vector<rvsdg::GammaNode::EntryVar> deadEntryVars;
  for (const auto & entryVar : gammaNode.GetEntryVars())
  {
    const bool isAlive = std::any_of(
        entryVar.branchArgument.begin(),
        entryVar.branchArgument.end(),
        [this](const rvsdg::Output * arg)
        {
          return Context_->isAlive(*arg);
        });
    if (!isAlive)
    {
      deadEntryVars.push_back(entryVar);
    }
  }
  gammaNode.RemoveEntryVars(deadEntryVars);
}

void
DeadNodeElimination::sweepTheta(rvsdg::ThetaNode & thetaNode) const
{
  // Determine dead loop variables.
  std::vector<rvsdg::ThetaNode::LoopVar> loopVars;
  for (const auto & loopVar : thetaNode.GetLoopVars())
  {
    if (!Context_->isAlive(*loopVar.pre) && !Context_->isAlive(*loopVar.output))
    {
      loopVar.post->divert_to(loopVar.pre);
      loopVars.push_back(loopVar);
    }
  }

  // Now that the loop variables to be eliminated only point to
  // their own pre-iteration values, any outputs within the subregion
  // that only contributed to computing the post-iteration values
  // of the variables are unlinked and can be removed as well.
  sweepRegion(*thetaNode.subregion());

  // There are now no other users of the pre-iteration values of the
  // variables to be removed left in the subregion anymore.
  // The variables have become "loop-invariant" and can simply
  // be eliminated from the theta node.
  thetaNode.RemoveLoopVars(std::move(loopVars));
}

void
DeadNodeElimination::sweepLambda(rvsdg::LambdaNode & lambdaNode) const
{
  sweepRegion(*lambdaNode.subregion());
  lambdaNode.PruneLambdaInputs();
}

void
DeadNodeElimination::sweepPhi(rvsdg::PhiNode & phiNode) const
{
  std::vector<rvsdg::PhiNode::FixVar> deadFixVars;
  std::vector<rvsdg::PhiNode::ContextVar> deadCtxVars;

  for (const auto & fixVar : phiNode.GetFixVars())
  {
    if (!Context_->isAlive(*fixVar.output) && !Context_->isAlive(*fixVar.recref))
    {
      deadFixVars.push_back(fixVar);
      // Temporarily redirect the variable so it refers to itself
      // (so the object is simply defined to be "itself").
      fixVar.result->divert_to(fixVar.recref);
    }
  }

  sweepRegion(*phiNode.subregion());

  for (const auto & ctxvar : phiNode.GetContextVars())
  {
    if (ctxvar.inner->IsDead())
    {
      deadCtxVars.push_back(ctxvar);
    }
  }

  phiNode.RemoveContextVars(std::move(deadCtxVars));
  phiNode.RemoveFixVars(std::move(deadFixVars));
}

void
DeadNodeElimination::sweepDelta(rvsdg::DeltaNode & deltaNode)
{
  // A delta subregion can only contain simple nodes. Thus, a simple prune is sufficient.
  deltaNode.subregion()->prune(false);

  deltaNode.PruneDeltaInputs();
}

void
DeadNodeElimination::removeNode(rvsdg::Node & node)
{
  if (is<LoadNonVolatileOperation>(node.GetOperation()))
  {
    for (auto & memoryStateOutput : LoadOperation::MemoryStateOutputs(node))
    {
      const auto origin = LoadOperation::MapMemoryStateOutputToInput(memoryStateOutput).origin();
      memoryStateOutput.divert_users(origin);
    }
    JLM_ASSERT(LoadOperation::LoadedValueOutput(node).IsDead());
  }

  remove(&node);
}

}

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
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

#include <typeindex>

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
  MarkAlive(const jlm::rvsdg::Output & output)
  {
    if (auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
    {
      SimpleNodes_.insert(simpleNode);
      return;
    }

    Outputs_.insert(&output);
  }

  bool
  IsAlive(const jlm::rvsdg::Output & output) const noexcept
  {
    if (auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
    {
      return SimpleNodes_.Contains(simpleNode);
    }

    return Outputs_.Contains(&output);
  }

  bool
  IsAlive(const rvsdg::Node & node) const noexcept
  {
    if (auto simpleNode = dynamic_cast<const jlm::rvsdg::SimpleNode *>(&node))
    {
      return SimpleNodes_.Contains(simpleNode);
    }

    for (size_t n = 0; n < node.noutputs(); n++)
    {
      if (IsAlive(*node.output(n)))
      {
        return true;
      }
    }

    return false;
  }

  static std::unique_ptr<Context>
  Create()
  {
    return std::make_unique<Context>();
  }

private:
  util::HashSet<const jlm::rvsdg::SimpleNode *> SimpleNodes_;
  util::HashSet<const jlm::rvsdg::Output *> Outputs_;
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
      : util::Statistics(Statistics::Id::DeadNodeElimination, sourceFile)
  {}

  void
  StartMarkStatistics(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(&graph.GetRootRegion()));
    AddMeasurement(Label::NumRvsdgInputsBefore, rvsdg::ninputs(&graph.GetRootRegion()));
    AddTimer(MarkTimerLabel_).start();
  }

  void
  StopMarkStatistics() noexcept
  {
    GetTimer(MarkTimerLabel_).stop();
  }

  void
  StartSweepStatistics() noexcept
  {
    AddTimer(SweepTimerLabel_).start();
  }

  void
  StopSweepStatistics(const rvsdg::Graph & graph) noexcept
  {
    GetTimer(SweepTimerLabel_).stop();
    AddMeasurement(Label::NumRvsdgNodesAfter, rvsdg::nnodes(&graph.GetRootRegion()));
    AddMeasurement(Label::NumRvsdgInputsAfter, rvsdg::ninputs(&graph.GetRootRegion()));
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
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
  Context_ = Context::Create();

  MarkRegion(region);
  SweepRegion(region);

  // Discard internal state to free up memory after we are done
  Context_.reset();
}

void
DeadNodeElimination::Run(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  Context_ = Context::Create();

  auto & rvsdg = module.Rvsdg();
  auto statistics = Statistics::Create(module.SourceFilePath().value());
  statistics->StartMarkStatistics(rvsdg);
  MarkRegion(rvsdg.GetRootRegion());
  statistics->StopMarkStatistics();

  statistics->StartSweepStatistics();
  SweepRvsdg(rvsdg);
  statistics->StopSweepStatistics(rvsdg);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Discard internal state to free up memory after we are done
  Context_.reset();
}

void
DeadNodeElimination::MarkRegion(const rvsdg::Region & region)
{
  for (size_t n = 0; n < region.nresults(); n++)
  {
    MarkOutput(*region.result(n)->origin());
  }
}

void
DeadNodeElimination::MarkOutput(const jlm::rvsdg::Output & output)
{
  if (Context_->IsAlive(output))
  {
    return;
  }

  Context_->MarkAlive(output);

  if (is<rvsdg::GraphImport>(&output))
  {
    return;
  }

  if (auto gamma = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(output))
  {
    MarkOutput(*gamma->predicate()->origin());
    for (const auto & result : gamma->MapOutputExitVar(output).branchResult)
    {
      MarkOutput(*result->origin());
    }
    return;
  }

  if (auto gamma = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(output))
  {
    auto external_origin = std::visit(
        [](const auto & rolevar) -> rvsdg::Output *
        {
          return rolevar.input->origin();
        },
        gamma->MapBranchArgument(output));
    MarkOutput(*external_origin);
    return;
  }

  if (auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(output))
  {
    auto loopvar = theta->MapOutputLoopVar(output);
    MarkOutput(*theta->predicate()->origin());
    MarkOutput(*loopvar.post->origin());
    MarkOutput(*loopvar.input->origin());
    return;
  }

  if (auto theta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    auto loopvar = theta->MapPreLoopVar(output);
    MarkOutput(*loopvar.output);
    MarkOutput(*loopvar.input->origin());
    return;
  }

  if (auto lambda = rvsdg::TryGetOwnerNode<rvsdg::LambdaNode>(output))
  {
    for (auto & result : lambda->GetFunctionResults())
    {
      MarkOutput(*result->origin());
    }
    return;
  }

  if (auto lambda = rvsdg::TryGetRegionParentNode<rvsdg::LambdaNode>(output))
  {
    if (auto ctxvar = lambda->MapBinderContextVar(output))
    {
      // Bound context variable.
      MarkOutput(*ctxvar->input->origin());
      return;
    }
    else
    {
      // Function argument.
      return;
    }
  }

  if (auto phi = rvsdg::TryGetOwnerNode<rvsdg::PhiNode>(output))
  {
    MarkOutput(*phi->MapOutputFixVar(output).result->origin());
    return;
  }

  if (auto phi = rvsdg::TryGetRegionParentNode<rvsdg::PhiNode>(output))
  {
    auto var = phi->MapArgument(output);
    if (auto fix = std::get_if<rvsdg::PhiNode::FixVar>(&var))
    {
      // Recursion argument
      MarkOutput(*fix->result->origin());
      return;
    }
    else if (auto ctx = std::get_if<rvsdg::PhiNode::ContextVar>(&var))
    {
      // Bound context variable.
      MarkOutput(*ctx->input->origin());
      return;
    }
    else
    {
      JLM_UNREACHABLE("Phi argument must be either fixpoint or context variable");
    }
  }

  if (const auto deltaNode = rvsdg::TryGetOwnerNode<rvsdg::DeltaNode>(output))
  {
    const auto result = deltaNode->subregion()->result(0);
    MarkOutput(*result->origin());
    return;
  }

  if (rvsdg::TryGetRegionParentNode<rvsdg::DeltaNode>(output))
  {
    const auto argument = util::assertedCast<const rvsdg::RegionArgument>(&output);
    MarkOutput(*argument->input()->origin());
    return;
  }

  if (const auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
  {
    for (size_t n = 0; n < simpleNode->ninputs(); n++)
    {
      MarkOutput(*simpleNode->input(n)->origin());
    }
    return;
  }

  JLM_UNREACHABLE("We should have never reached this statement.");
}

void
DeadNodeElimination::SweepRvsdg(rvsdg::Graph & rvsdg) const
{
  SweepRegion(rvsdg.GetRootRegion());

  // Remove dead imports
  for (size_t n = rvsdg.GetRootRegion().narguments() - 1; n != static_cast<size_t>(-1); n--)
  {
    if (!Context_->IsAlive(*rvsdg.GetRootRegion().argument(n)))
    {
      rvsdg.GetRootRegion().RemoveArgument(n);
    }
  }
}

void
DeadNodeElimination::SweepRegion(rvsdg::Region & region) const
{
  region.prune(false);

  std::vector<std::vector<rvsdg::Node *>> nodesTopDown(region.numNodes());
  for (auto & node : region.Nodes())
  {
    nodesTopDown[node.depth()].push_back(&node);
  }

  for (auto it = nodesTopDown.rbegin(); it != nodesTopDown.rend(); it++)
  {
    for (auto node : *it)
    {
      if (!Context_->IsAlive(*node))
      {
        remove(node);
        continue;
      }

      if (auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(node))
      {
        SweepStructuralNode(*structuralNode);
      }
    }
  }

  JLM_ASSERT(region.numBottomNodes() == 0);
}

void
DeadNodeElimination::SweepStructuralNode(rvsdg::StructuralNode & node) const
{
  rvsdg::MatchTypeOrFail(
      node,
      [this](rvsdg::GammaNode & node)
      {
        SweepGamma(node);
      },
      [this](rvsdg::ThetaNode & node)
      {
        SweepTheta(node);
      },
      [this](rvsdg::LambdaNode & node)
      {
        SweepLambda(node);
      },
      [this](rvsdg::PhiNode & node)
      {
        SweepPhi(node);
      },
      [](rvsdg::DeltaNode & node)
      {
        SweepDelta(node);
      });
}

void
DeadNodeElimination::SweepGamma(rvsdg::GammaNode & gammaNode) const
{
  // Remove dead exit vars.
  std::vector<rvsdg::GammaNode::ExitVar> deadExitVars;
  for (const auto & exitvar : gammaNode.GetExitVars())
  {
    if (!Context_->IsAlive(*exitvar.output))
    {
      deadExitVars.push_back(exitvar);
    }
  }
  gammaNode.RemoveExitVars(deadExitVars);

  // Sweep gamma subregions
  for (size_t r = 0; r < gammaNode.nsubregions(); r++)
  {
    SweepRegion(*gammaNode.subregion(r));
  }

  // Remove dead entry vars.
  std::vector<rvsdg::GammaNode::EntryVar> deadEntryVars;
  for (const auto & entryvar : gammaNode.GetEntryVars())
  {
    bool alive = std::any_of(
        entryvar.branchArgument.begin(),
        entryvar.branchArgument.end(),
        [this](const rvsdg::Output * arg)
        {
          return Context_->IsAlive(*arg);
        });
    if (!alive)
    {
      deadEntryVars.push_back(entryvar);
    }
  }
  gammaNode.RemoveEntryVars(deadEntryVars);
}

void
DeadNodeElimination::SweepTheta(rvsdg::ThetaNode & thetaNode) const
{
  // Determine loop variables to be removed.
  std::vector<rvsdg::ThetaNode::LoopVar> loopvars;
  for (const auto & loopvar : thetaNode.GetLoopVars())
  {
    if (!Context_->IsAlive(*loopvar.pre) && !Context_->IsAlive(*loopvar.output))
    {
      loopvar.post->divert_to(loopvar.pre);
      loopvars.push_back(loopvar);
    }
  }

  // Now that the loop variables to be eliminated only point to
  // their own pre-iteration values, any outputs within the subregion
  // that only contributed to computing the post-iteration values
  // of the variables are unlinked and can be removed as well.
  SweepRegion(*thetaNode.subregion());

  // There are now no other users of the pre-iteration values of the
  // variables to be removed left in the subregion anymore.
  // The variables have become "loop-invariant" and can simply
  // be eliminated from the theta node.
  thetaNode.RemoveLoopVars(std::move(loopvars));
}

void
DeadNodeElimination::SweepLambda(rvsdg::LambdaNode & lambdaNode) const
{
  SweepRegion(*lambdaNode.subregion());
  lambdaNode.PruneLambdaInputs();
}

void
DeadNodeElimination::SweepPhi(rvsdg::PhiNode & phiNode) const
{
  std::vector<rvsdg::PhiNode::FixVar> deadFixvars;
  std::vector<rvsdg::PhiNode::ContextVar> deadCtxvars;

  for (const auto & fixvar : phiNode.GetFixVars())
  {
    bool isDead = !Context_->IsAlive(*fixvar.output) && !Context_->IsAlive(*fixvar.recref);
    if (isDead)
    {
      deadFixvars.push_back(fixvar);
      // Temporarily redirect the variable so it refers to itself
      // (so the object is simply defined to be "itself").
      fixvar.result->divert_to(fixvar.recref);
    }
  }

  SweepRegion(*phiNode.subregion());

  for (const auto & ctxvar : phiNode.GetContextVars())
  {
    if (ctxvar.inner->IsDead())
    {
      deadCtxvars.push_back(ctxvar);
    }
  }

  phiNode.RemoveContextVars(std::move(deadCtxvars));
  phiNode.RemoveFixVars(std::move(deadFixvars));
}

void
DeadNodeElimination::SweepDelta(rvsdg::DeltaNode & deltaNode)
{
  // A delta subregion can only contain simple nodes. Thus, a simple prune is sufficient.
  deltaNode.subregion()->prune(false);

  deltaNode.PruneDeltaInputs();
}

}

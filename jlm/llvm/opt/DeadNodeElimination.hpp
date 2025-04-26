/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_DEADNODEELIMINATION_HPP
#define JLM_LLVM_OPT_DEADNODEELIMINATION_HPP

#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/Transformation.hpp>

#include <typeindex>

namespace jlm::rvsdg
{
class GammaNode;
class Graph;
class output;
class StructuralNode;
class ThetaNode;
class Region;
}

namespace jlm::llvm
{

namespace delta
{
class node;
}

class LambdaNode;

namespace phi
{
class node;
}

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
class DNEContext final
{
public:
  void
  MarkAlive(const rvsdg::output & output)
  {
    if (const auto simpleOutput = dynamic_cast<const rvsdg::SimpleOutput *>(&output))
    {
      SimpleNodes_.Insert(simpleOutput->node());
      return;
    }

    Outputs_.Insert(&output);
  }

  bool
  IsAlive(const rvsdg::output & output) const noexcept
  {
    if (const auto simpleOutput = dynamic_cast<const rvsdg::SimpleOutput *>(&output))
    {
      return SimpleNodes_.Contains(simpleOutput->node());
    }

    return Outputs_.Contains(&output);
  }

  bool
  IsAlive(const rvsdg::Node & node) const noexcept
  {
    if (const auto simpleNode = dynamic_cast<const rvsdg::SimpleNode *>(&node))
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

private:
  util::HashSet<const rvsdg::SimpleNode *> SimpleNodes_;
  util::HashSet<const rvsdg::output *> Outputs_;
};

class DNEStructuralNodeHandler
{
public:
  virtual ~DNEStructuralNodeHandler();

  virtual std::type_index
  GetTypeInfo() const = 0;

  virtual std::optional<std::vector<rvsdg::output *>>
  ComputeMarkPhaseContinuations(const rvsdg::output & output) const = 0;

  virtual void
  SweepNodeEntry(rvsdg::StructuralNode & structuralNode, const DNEContext & context) const = 0;

  virtual void
  SweepNodeExit(rvsdg::StructuralNode & structuralNode, const DNEContext & context) const = 0;
};

class DNEGammaNodeHandler final : public DNEStructuralNodeHandler
{
public:
  ~DNEGammaNodeHandler() override;

  DNEGammaNodeHandler(const DNEGammaNodeHandler &) = delete;

  DNEGammaNodeHandler &
  operator=(const DNEGammaNodeHandler &) = delete;

  DNEGammaNodeHandler(DNEGammaNodeHandler &&) = delete;

  DNEGammaNodeHandler &
  operator=(DNEGammaNodeHandler &&) = delete;

  std::type_index
  GetTypeInfo() const override;

  std::optional<std::vector<rvsdg::output *>>
  ComputeMarkPhaseContinuations(const rvsdg::output & output) const override;

  void
  SweepNodeEntry(rvsdg::StructuralNode & structuralNode, const DNEContext & context) const override;

  void
  SweepNodeExit(rvsdg::StructuralNode & structuralNode, const DNEContext & context) const override;

  static DNEStructuralNodeHandler *
  GetInstance();

private:
  DNEGammaNodeHandler();
};

/** \brief Dead Node Elimination Optimization
 *
 * Dead Node Elimination removes all nodes that do not contribute to the result of a computation. A
 * node is considered dead if all its outputs are dead, and an output is considered dead if it has
 * no users or all its users are already dead. An input (and therefore an outputs' user) is
 * considered dead if the corresponding node is dead. We call all nodes, inputs, and outputs that
 * are not dead alive.
 *
 * The Dead Node Elimination optimization consists of two phases: mark and sweep. The mark phase
 * traverses the RVSDG and marks all nodes, inputs, and outputs that it finds as alive, while the
 * sweep phase removes then all nodes, inputs, and outputs that were not discovered by the mark
 * phase, i.e., all dead nodes, inputs, and outputs.
 *
 * Please see TestDeadNodeElimination.cpp for Dead Node Elimination examples.
 */
class DeadNodeElimination final : public rvsdg::Transformation
{
  class Statistics;

public:
  ~DeadNodeElimination() noexcept override;

  explicit DeadNodeElimination(std::vector<const DNEStructuralNodeHandler *> handlers);

  DeadNodeElimination(const DeadNodeElimination &) = delete;

  DeadNodeElimination(DeadNodeElimination &&) = delete;

  DeadNodeElimination &
  operator=(const DeadNodeElimination &) = delete;

  DeadNodeElimination &
  operator=(DeadNodeElimination &&) = delete;

  void
  run(rvsdg::Region & region);

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
  void
  MarkRegion(const rvsdg::Region & region);

  void
  MarkOutput(const jlm::rvsdg::output & output);

  void
  SweepRvsdg(rvsdg::Graph & rvsdg) const;

  void
  SweepRegion(rvsdg::Region & region) const;

  void
  SweepStructuralNode(rvsdg::StructuralNode & node) const;

  void
  SweepTheta(rvsdg::ThetaNode & thetaNode) const;

  void
  SweepLambda(rvsdg::LambdaNode & lambdaNode) const;

  void
  SweepPhi(phi::node & phiNode) const;

  static void
  SweepDelta(delta::node & deltaNode);

  DNEContext Context_;
  std::vector<const DNEStructuralNodeHandler *> Handlers_;
};

}

#endif

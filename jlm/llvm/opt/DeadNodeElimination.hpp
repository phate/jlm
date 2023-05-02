/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_DEADNODEELIMINATION_HPP
#define JLM_LLVM_OPT_DEADNODEELIMINATION_HPP

#include <jlm/llvm/opt/optimization.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/structural-node.hpp>

namespace jive {
class gamma_node;

class theta_node;
}

namespace jlm {

namespace delta {
class node;
}
namespace lambda {
class node;
}

namespace phi { class node; }

class RvsdgModule;

/** \brief Dead Node Elimination Optimization
 *
 * Dead Node Elimination removes all nodes that do not contribute to the result of a computation. A node is considered
 * dead if all its outputs are dead, and an output is considered dead if it has no users or all its users are already
 * dead. An input (and therefore an outputs' user) is considered dead if the corresponding node is dead. We call all
 * nodes, inputs, and outputs that are not dead alive.
 *
 * The Dead Node Elimination optimization consists of two phases: mark and sweep. The mark phase traverses the RVSDG and
 * marks all nodes, inputs, and outputs that it finds as alive, while the sweep phase removes then all nodes, inputs,
 * and outputs that were not discovered by the mark phase, i.e., all dead nodes, inputs, and outputs.
 *
 * Please see TestDeadNodeElimination.cpp for Dead Node Elimination examples.
 */
class DeadNodeElimination final : public optimization {

  /** \brief Dead Node Elimination context class
   *
   * This class keeps track of all the nodes and outputs that are alive. In contrast to all other nodes, a simple node
   * is considered alive if already a single of its outputs is alive. For this reason, this class keeps separately track
   * of simple nodes and therefore avoids to store all its outputs (and instead stores the node itself).
   * By marking the entire node as alive, we also avoid that we reiterate through all inputs of this node again in the
   * future. The following example illustrates the issue:
   *
   * o1 ... oN = Node2 i1 ... iN
   * p1 ... pN = Node1 o1 ... oN
   *
   * When we mark o1 as alive, we actually mark the entire Node2 as alive. This means that when we try to mark o2 alive
   * in the future, we can immediately stop marking instead of reiterating through i1 ... iN again. Thus, by marking the
   * entire simple node instead of just its outputs, we reduce the runtime for marking Node2 from
   * O(oN x iN) to O(oN + iN).
   */
  class Context final {
  public:
    void
    MarkAlive(const jive::output & output)
    {
      if (auto simpleOutput = dynamic_cast<const jive::simple_output*>(&output)) {
          simpleNodes_.insert(simpleOutput->node());
          return;
      }

      outputs_.insert(&output);
    }

    bool
    IsAlive(const jive::output & output) const noexcept
    {
      if (auto simpleOutput = dynamic_cast<const jive::simple_output*>(&output))
        return simpleNodes_.find(simpleOutput->node()) != simpleNodes_.end();

      return outputs_.find(&output) != outputs_.end();
    }

    bool
    IsAlive(const jive::node & node) const noexcept
    {
      if (auto simpleNode = dynamic_cast<const jive::simple_node*>(&node))
        return simpleNodes_.find(simpleNode) != simpleNodes_.end();

      for (size_t n = 0; n < node.noutputs(); n++) {
        if (IsAlive(*node.output(n)))
          return true;
      }

      return false;
    }

    void
    Clear()
    {
      simpleNodes_.clear();
      outputs_.clear();
    }

  private:
    std::unordered_set<const jive::simple_node*> simpleNodes_;
    std::unordered_set<const jive::output*> outputs_;
  };

  class Statistics;

public:
	~DeadNodeElimination() override;

	void
	run(jive::region & region);

	void
	run(
    RvsdgModule & module,
    StatisticsCollector & statisticsCollector) override;

private:
  void
  ResetState();

  void
  Mark(const jive::region & region);

  void
  Mark(const jive::output & output);

  void
  Sweep(jive::graph & graph) const;

  void
  Sweep(jive::region & region) const;

  void
  Sweep(jive::structural_node & node) const;

  void
  SweepGamma(jive::gamma_node & gammaNode) const;

  void
  SweepTheta(jive::theta_node & thetaNode) const;

  void
  SweepLambda(lambda::node & lambdaNode) const;

  void
  SweepPhi(phi::node & phiNode) const;

  void
  SweepDelta(delta::node & deltaNode) const;

  Context context_;
};

}

#endif

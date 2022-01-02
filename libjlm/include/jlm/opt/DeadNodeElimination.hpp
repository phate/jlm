/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_DEADNODEELIMINATION_HPP
#define JLM_OPT_DEADNODEELIMINATION_HPP

#include <jlm/opt/optimization.hpp>

#include <jive/rvsdg/simple-node.hpp>
#include <jive/rvsdg/structural-node.hpp>

namespace jlm {

class rvsdg_module;
class StatisticsDescriptor;

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
   * This class keeps track of all the outputs that are alive.
   */
  class Context final {
  public:
    void
    Mark(const jive::output & output)
    {
      outputs_.insert(&output);
    }

    bool
    IsAlive(const jive::output & output) const noexcept
    {
      return outputs_.find(&output) != outputs_.end();
    }

    bool
    IsAlive(const jive::node & node) const noexcept
    {
      for (size_t n = 0; n < node.noutputs(); n++) {
        if (IsAlive(*node.output(n)))
          return true;
      }

      return false;
    }

    void
    Clear()
    {
      outputs_.clear();
    }

  private:
    std::unordered_set<const jive::output*> outputs_;
  };

  class Statistics;

public:
	~DeadNodeElimination() override;

	void
	run(jive::region & region);

	void
	run(
    rvsdg_module & module,
    const StatisticsDescriptor & sd) override;

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
  Sweep(jive::simple_node & node) const;

  void
  Sweep(jive::structural_node & node) const;

  void
  SweepGamma(jive::structural_node & node) const;

  void
  SweepTheta(jive::structural_node & node) const;

  void
  SweepLambda(jive::structural_node & node) const;

  void
  SweepPhi(jive::structural_node & node) const;

  void
  SweepDelta(jive::structural_node & node) const;

  Context context_;
};

}

#endif

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

/** \brief Dead Node Elimination
*
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

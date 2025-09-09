/*
 * Copyright 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_THETA_HPP
#define JLM_RVSDG_THETA_HPP

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/util/HashSet.hpp>

#include <optional>

namespace jlm::rvsdg
{

class ThetaOperation final : public StructuralOperation
{
public:
  ~ThetaOperation() noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;
};

class ThetaNode final : public StructuralNode
{
public:
  ~ThetaNode() noexcept override;

private:
  explicit ThetaNode(rvsdg::Region & parent);

public:
  /**
   * \brief Description of a loop-carried variable.
   *
   * A loop-carried variable from the POV of a theta node has
   * multiple representations (entry, pre-iteration,
   * post-iteration, exit). This structure bundles
   * all representations of a single loop-carried variable.
   */
  struct LoopVar
  {
    /**
     * \brief Variable at loop entry (input to theta).
     */
    rvsdg::Input * input;
    /**
     * \brief Variable before iteration (input argument to subregion).
     */
    rvsdg::Output * pre;
    /**
     * \brief Variable after iteration (output result from subregion).
     */
    rvsdg::Input * post;
    /**
     * \brief Variable at loop exit (output of theta).
     */
    rvsdg::Output * output;
  };

  [[nodiscard]] const ThetaOperation &
  GetOperation() const noexcept override;

  static ThetaNode *
  create(rvsdg::Region * parent)
  {
    return new ThetaNode(*parent);
  }

  [[nodiscard]] rvsdg::Region *
  subregion() const noexcept
  {
    return StructuralNode::subregion(0);
  }

  [[nodiscard]] RegionResult *
  predicate() const noexcept
  {
    auto result = subregion()->result(0);
    JLM_ASSERT(is<const ControlType>(result->Type()));
    return result;
  }

  inline void
  set_predicate(jlm::rvsdg::Output * p)
  {
    auto node = TryGetOwnerNode<Node>(*predicate()->origin());

    predicate()->divert_to(p);
    if (node && node->IsDead())
      remove(node);
  }

  /**
   * \brief Creates a new loop-carried variable.
   *
   * \param origin
   *   Input value at start of loop.
   *
   * \returns
   *   Loop variable description.
   *
   * Creates a new variable that is routed through the loop. The variable
   * is set up such that the post-iteration value is the same as the
   * pre-iteration value (i.e. the value remains unchanged through
   * the loop). Caller can redirect edges inside the loop to turn this
   * into a variable changed by the loop
   */
  LoopVar
  AddLoopVar(rvsdg::Output * origin);

  /**
   * \brief Removes loop variables.
   *
   * \param loopvars
   *   The loop variables to be removed.
   *
   * \pre
   *   For each loopvar in \p loopvars the following must hold:
   *   - loopvar.pre->origin() == loopvar.post
   *   - loopvar.pre has no other users besides loopvar.post
   *   - loopvar.output has no users
   *
   * Removes loop variables from this theta construct. The
   * loop variables must be loop-invariant and otherwise unused.
   * See dead node elimination that is explicitly structured
   * to restructure loops before processing to ensure this
   * invariant.
   */
  void
  RemoveLoopVars(std::vector<LoopVar> loopvars);

  ThetaNode *
  copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const override;

  /**
   * \brief Maps variable at entry to full varibale description.
   *
   * \param input
   *   Input to the theta node.
   *
   * \returns
   *   The loop variable description.
   *
   * \pre
   *   \p input must be an input to this node.
   *
   * Returns the full description of the loop variable corresponding
   * to this entry into the theta node.
   */
  [[nodiscard]] LoopVar
  MapInputLoopVar(const rvsdg::Input & input) const;

  /**
   * \brief A safe wrapper for the MapInputLoopVar function.
   *
   * \returns
   *   An std::optional with the loop variable.
   */
  std::optional<LoopVar>
  TryMapInputLoopVar(const rvsdg::Input * input) const
  {
    if (rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*input) != this)
    {
      return std::nullopt;
    }
    return this->MapInputLoopVar(*input);
  };

  /**
   * \brief Maps variable at start of loop iteration to full varibale description.
   *
   * \param argument
   *   Argument of theta region.
   *
   * \returns
   *   The loop variable description.
   *
   * \pre
   *   \p argument must be an argument to the subregion of this node.
   *
   * Returns the full description of the loop variable corresponding
   * to this variable at the start of each loop iteration.
   */
  [[nodiscard]] LoopVar
  MapPreLoopVar(const rvsdg::Output & argument) const;

  /**
   * \brief Maps variable at end of loop iteration to full varibale description.
   *
   * \param result
   *   Result of theta region.
   *
   * \returns
   *   The loop variable description.
   *
   * \pre
   *   \p result must be a result to the subregion of this node.
   *
   * Returns the full description of the loop variable corresponding
   * to this variable at the end of each loop iteration.
   */
  [[nodiscard]] LoopVar
  MapPostLoopVar(const rvsdg::Input & result) const;

  /**
   * \brief Maps variable at exit to full varibale description.
   *
   * \param output
   *   Output of this theta node
   *
   * \returns
   *   The loop variable description
   *
   * \pre
   *   \p output must be an output of this node
   *
   * Returns the full description of the loop variable corresponding
   * to this loop exit value.
   */
  [[nodiscard]] LoopVar
  MapOutputLoopVar(const rvsdg::Output & output) const;

  /**
   * \brief Returns all loop variables.
   *
   * \returns
   *   List of loop variable descriptions.
   */
  [[nodiscard]] std::vector<LoopVar>
  GetLoopVars() const;
};

static inline bool
ThetaLoopVarIsInvariant(const ThetaNode::LoopVar & loopVar) noexcept
{
  return loopVar.post->origin() == loopVar.pre;
}

}

#endif

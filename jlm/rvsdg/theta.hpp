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

  virtual std::string
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
    rvsdg::input * input;
    /**
     * \brief Variable before iteration (input argument to subregion).
     */
    rvsdg::output * pre;
    /**
     * \brief Variable after iteration (output result from subregion).
     */
    rvsdg::input * post;
    /**
     * \brief Variable at loop exit (output of theta).
     */
    rvsdg::output * output;
  };

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
    JLM_ASSERT(dynamic_cast<const ControlType *>(&result->type()));
    return result;
  }

  inline void
  set_predicate(jlm::rvsdg::output * p)
  {
    auto node = output::GetNode(*predicate()->origin());

    predicate()->divert_to(p);
    if (node && !node->has_users())
      remove(node);
  }

  /**
   * Remove theta outputs and their respective results.
   *
   * An output must match the condition specified by \p match and it must be dead.
   *
   * @tparam F A type that supports the function call operator: bool operator(const ThetaOutput&)
   * @param match Defines the condition of the elements to remove.
   * @return The inputs corresponding to the removed outputs.
   *
   * \note The application of this method might leave the theta node in an invalid state. Some
   * inputs might refer to outputs that have been removed by the application of this method. It
   * is up to the caller to ensure that the invariants of the theta node will eventually be met
   * again.
   *
   * \see RemoveThetaInputsWhere()
   * \see ThetaOutput#IsDead()
   */
  template<typename F>
  util::HashSet<const rvsdg::input *>
  RemoveThetaOutputsWhere(const F & match);

  /**
   * Remove all dead theta outputs and their respective results.
   *
   * @return The inputs corresponding to the removed outputs.
   *
   * \note The application of this method might leave the theta node in an invalid state. Some
   * inputs might refer to outputs that have been removed by the application of this method. It
   * is up to the caller to ensure that the invariants of the theta node will eventually be met
   * again.
   *
   * \see RemoveThetaOutputsWhere()
   * \see ThetaOutput#IsDead()
   */
  util::HashSet<const rvsdg::input *>
  PruneThetaOutputs()
  {
    auto match = [](const rvsdg::output &)
    {
      return true;
    };

    return RemoveThetaOutputsWhere(match);
  }

  /**
   * Remove theta inputs and their respective arguments.
   *
   * An input must match the condition specified by \p match and its respective argument must be
   * dead.
   *
   * @tparam F A type that supports the function call operator: bool operator(const jlm::input&)
   * @param match Defines the condition of the elements to remove.
   * @return The outputs corresponding to the removed outputs.
   *
   * \note The application of this method might leave the theta node in an invalid state. Some
   * outputs might refer to inputs that have been removed by the application of this method. It
   * is up to the caller to ensure that the invariants of the theta node will eventually be met
   * again.
   *
   * \see RemoveThetaOutputsWhere()
   * \see RegionArgument#IsDead()
   */
  template<typename F>
  util::HashSet<const rvsdg::output *>
  RemoveThetaInputsWhere(const F & match);

  /**
   * Remove all dead theta inputs and their respective arguments.
   *
   * @return The outputs corresponding to the removed outputs.
   *
   * \note The application of this method might leave the theta node in an invalid state. Some
   * outputs might refer to inputs that have been removed by the application of this method. It
   * is up to the caller to ensure that the invariants of the theta node will eventually be met
   * again.
   *
   * \see RemoveThetaInputsWhere()
   * \see RegionArgument#IsDead()
   */
  util::HashSet<const rvsdg::output *>
  PruneThetaInputs()
  {
    auto match = [](const rvsdg::input &)
    {
      return true;
    };

    return RemoveThetaInputsWhere(match);
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
  AddLoopVar(rvsdg::output * origin);

  virtual ThetaNode *
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
  MapInputLoopVar(const rvsdg::input & input) const;

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
  MapPreLoopVar(const rvsdg::output & argument) const;

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
  MapPostLoopVar(const rvsdg::input & result) const;

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
  MapOutputLoopVar(const rvsdg::output & output) const;

  /**
   * \brief Returns all loop variables.
   *
   * \returns
   *   List of loop variable descriptions.
   */
  [[nodiscard]] std::vector<LoopVar>
  GetLoopVars() const;

private:
  // Calling RemoveThetaInputsWhere/RemoveThetaOutputsWhere can result
  // in inputs (and pre-loop arguments) and outputs (and post-loop results)
  // to become unmatched. In this case, the theta node itself has
  // "invalid" shape until fixed properly.
  // The indices of unmatched inputs/outputs are tracked here to
  // detect this situation, and also to provide correct mapping.
  // Computing the mapping is a bit fiddly as it requires adjusting
  // indices accordingly, should seriously consider whether this
  // is really necessary or things can rather be reformulated such that
  // inputs/outputs are always consistent.

  std::optional<std::size_t>
  MapInputToOutputIndex(std::size_t index) const noexcept;

  std::optional<std::size_t>
  MapOutputToInputIndex(std::size_t index) const noexcept;

  void
  MarkInputIndexErased(std::size_t index) noexcept;

  void
  MarkOutputIndexErased(std::size_t index) noexcept;

  std::vector<std::size_t> unmatchedInputs;
  std::vector<std::size_t> unmatchedOutputs;
};

static inline bool
ThetaLoopVarIsInvariant(const ThetaNode::LoopVar & loopVar) noexcept
{
  return loopVar.post->origin() == loopVar.pre;
}

/* theta node method definitions */

template<typename F>
util::HashSet<const rvsdg::input *>
ThetaNode::RemoveThetaOutputsWhere(const F & match)
{
  util::HashSet<const rvsdg::input *> deadInputs;

  auto loopvars = GetLoopVars();
  // iterate backwards to avoid the invalidation of 'n' by RemoveOutput()
  for (size_t n = loopvars.size(); n > 0; --n)
  {
    auto & loopvar = loopvars[n - 1];
    if (loopvar.output->IsDead() && match(*loopvar.output))
    {
      deadInputs.Insert(loopvar.input);
      subregion()->RemoveResult(loopvar.post->index());
      MarkOutputIndexErased(loopvar.output->index());
      RemoveOutput(loopvar.output->index());
    }
  }

  return deadInputs;
}

template<typename F>
util::HashSet<const rvsdg::output *>
ThetaNode::RemoveThetaInputsWhere(const F & match)
{
  util::HashSet<const rvsdg::output *> deadOutputs;

  // iterate backwards to avoid the invalidation of 'n' by RemoveInput()
  for (size_t n = ninputs() - 1; n != static_cast<size_t>(-1); n--)
  {
    auto & thetaInput = *input(n);
    auto loopvar = MapInputLoopVar(thetaInput);

    if (loopvar.pre->IsDead() && match(thetaInput))
    {
      deadOutputs.Insert(loopvar.output);
      subregion()->RemoveArgument(loopvar.pre->index());
      MarkInputIndexErased(thetaInput.index());
      RemoveInput(thetaInput.index());
    }
  }

  return deadOutputs;
}

}

#endif

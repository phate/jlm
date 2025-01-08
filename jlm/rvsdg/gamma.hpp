/*
 * Copyright 2010 2011 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_GAMMA_HPP
#define JLM_RVSDG_GAMMA_HPP

#include <optional>

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/structural-node.hpp>

namespace jlm::rvsdg
{

class output;
class Type;

class GammaOperation final : public StructuralOperation
{
public:
  ~GammaOperation() noexcept override;

  explicit constexpr GammaOperation(size_t nalternatives) noexcept
      : StructuralOperation(),
        nalternatives_(nalternatives)
  {}

  inline size_t
  nalternatives() const noexcept
  {
    return nalternatives_;
  }

  virtual std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  virtual bool
  operator==(const Operation & other) const noexcept override;

private:
  size_t nalternatives_;
};

/* gamma node */

class GammaNode : public StructuralNode
{
public:
  ~GammaNode() noexcept override;

private:
  GammaNode(rvsdg::output * predicate, size_t nalternatives);

public:
  /**
   * \brief A variable routed into all gamma regions.
   */
  struct EntryVar
  {
    /**
     * \brief Variable at entry point (input to gamma node).
     */
    rvsdg::input * input;
    /**
     * \brief Variable inside each of the branch regions (argument per subregion).
     */
    std::vector<rvsdg::output *> branchArgument;
  };

  /**
   * \brief A variable routed out of all gamma regions as result.
   */
  struct ExitVar
  {
    /**
     * \brief Variable exit points (results per subregion).
     */
    std::vector<rvsdg::input *> branchResult;
    /**
     * \brief Output of gamma.
     */
    rvsdg::output * output;
  };

  static GammaNode *
  create(jlm::rvsdg::output * predicate, size_t nalternatives)
  {
    return new GammaNode(predicate, nalternatives);
  }

  inline rvsdg::input *
  predicate() const noexcept;

  /**
   * \brief Routes a variable into the gamma branches.
   *
   * \param origin
   *   Value to be routed in.
   *
   * \returns
   *   Description of entry variable.
   *
   * Routes a variable into a gamma region. To access the
   * variable in each branch use \ref EntryVar::branchArgument.
   */
  EntryVar
  AddEntryVar(rvsdg::output * origin);

  /**
   * \brief Gets entry variable by index.
   *
   * \param index
   *   Index of entry variable
   *
   * \returns
   *   Description of entry variable.
   *
   * Looks up the \p index 'th entry variable into the gamma
   * node and returns its description.
   */
  EntryVar
  GetEntryVar(std::size_t index) const;

  /**
   * \brief Gets all entry variables for this gamma.
   */
  std::vector<EntryVar>
  GetEntryVars() const;

  /**
   * \brief Maps gamma input to entry variable.
   *
   * \param input
   *   Input to be mapped.
   *
   * \returns
   *   The entry variable description corresponding to this input
   *
   * \pre
   *   \p input must be an input of this node and must not be the predicate
   *
   * Maps the gamma input to the entry variable description corresponding
   * to it. This allows to trace the value through to users in the
   * gamma subregions.
   */
  EntryVar
  MapInputEntryVar(const rvsdg::input & input) const;

  /**
   * \brief Maps branch subregion entry argument to gamma entry variable.
   *
   * \param output
   *   The branch argument to be mapped.
   *
   * \returns
   *   The entry variable description corresponding to this input
   *
   * \pre
   *   \p output must be the entry argument to a subregion of this gamma nade.
   *
   * Maps the subregion entry argument to the entry variable description
   * corresponding to it. This allows to trace the value to users in other
   * branches as well as its def site preceding the gamma node:
   */
  EntryVar
  MapBranchArgumentEntryVar(const rvsdg::output & output) const;

  /**
   * \brief Routes per-branch result of gamma to output
   *
   * \param values
   *   Value to be routed out.
   *
   * \returns
   *   Description of exit variable.
   *
   * Routes per-branch values for a particular variable
   * out of the gamma regions and makes it available as
   * output of the gamma node.
   */
  ExitVar
  AddExitVar(std::vector<rvsdg::output *> values);

  /**
   * \brief Gets all exit variables for this gamma.
   */
  std::vector<ExitVar>
  GetExitVars() const;

  /**
   * \brief Maps gamma output to exit variable description.
   *
   * \param output
   *   Output to be mapped.
   *
   * \returns
   *   The exit variable description corresponding to this output.
   *
   * \pre
   *   \p output must be an output of this node.
   *
   * Maps the gamma output to the exit variable description corresponding
   * to it. This allows to trace the value through to users in the
   * gamma subregions.
   */
  ExitVar
  MapOutputExitVar(const rvsdg::output & output) const;

  /**
   * \brief Maps gamma region exit result to exit variable description.
   *
   * \param input
   *   The result to be mapped to be mapped.
   *
   * \returns
   *   The exit variable description corresponding to this output.
   *
   * \pre
   *   \p input must be a result of a subregion of this node.
   *
   * Maps the gamma region result to the exit variable description
   * corresponding to it.
   */
  ExitVar
  MapBranchResultExitVar(const rvsdg::input & input) const;

  /**
   * Removes all gamma outputs and their respective results. The outputs must have no users and
   * match the condition specified by \p match.
   *
   * @tparam F A type that supports the function call operator: bool operator(const rvsdg::output&)
   * @param match Defines the condition of the elements to remove.
   */
  template<typename F>
  void
  RemoveGammaOutputsWhere(const F & match);

  /**
   * Removes all outputs that have no users.
   */
  void
  PruneOutputs()
  {
    auto match = [](const rvsdg::output &)
    {
      return true;
    };

    RemoveGammaOutputsWhere(match);
  }

  virtual GammaNode *
  copy(jlm::rvsdg::Region * region, SubstitutionMap & smap) const override;
};

/**
 * \brief Determines whether a gamma exit var is path-invariant.
 *
 * \param gamma
 *   The gamma node which we are testing for.
 *
 * \param exitvar
 *   Exit variable of the gamma node.
 *
 * \returns
 *   The common (invariant) origin of this output, or nullopt.
 *
 * \pre
 *   \p exitvar must be an \ref GammaNode::ExitVar of \p gamma
 *
 * Checks whether the gamma effectively assigns the same input value to
 * this exit variable on all paths of the gamma. If this is the case, it
 * returns the origin of the common input.
 */
std::optional<rvsdg::output *>
GetGammaInvariantOrigin(const GammaNode & gamma, const GammaNode::ExitVar & exitvar);

/* gamma node method definitions */

inline rvsdg::input *
GammaNode::predicate() const noexcept
{
  return StructuralNode::input(0);
}

template<typename F>
void
GammaNode::RemoveGammaOutputsWhere(const F & match)
{
  // iterate backwards to avoid the invalidation of 'n' by RemoveOutput()
  for (size_t n = noutputs() - 1; n != static_cast<size_t>(-1); n--)
  {
    if (output(n)->nusers() == 0 && match(*output(n)))
    {
      for (size_t r = 0; r < nsubregions(); r++)
      {
        subregion(r)->RemoveResult(n);
      }

      RemoveOutput(n);
    }
  }
}

/**
 * Reduces a gamma node with a statically known predicate to the respective subregion determined
 * by the value of the predicate.
 *
 * c = gamma 0
 *   []
 *     x = 45
 *   [c <= x]
 *   []
 *     y = 37
 *   [c <= y]
 * ... = add c + 5
 * =>
 * c = 45
 * ... = add c + 5
 *
 * @param node A gamma node that is supposed to be reduced.
 * @return True, if transformation was successful, otherwise false.
 */
bool
ReduceGammaWithStaticallyKnownPredicate(Node & node);

/**
 * Reduces the predicate of a gamma node g1 from the constants that originate from another gamma
 * node g2 to the predicate of g2.
 *
 * p2 = gamma p1
 *  []
 *    x = 0
 *  [p2 <= x]
 *  []
 *    y = 1
 *  [p2 <= y]
 * ... = gamma p2
 * =>
 * p2 = gamma p1
 *  []
 *    x = 0
 *  [p2 <= x]
 *  []
 *    y = 1
 *  [p2 <= y]
 * ... = gamma p1
 *
 * @param node A gamma node that is supposed to be reduced.
 * @return True, if the transformation was successful, otherwise false.
 */
bool
ReduceGammaControlConstant(Node & node);

/**
 * Reduces all invariant variables of gamma node and diverts the users of the gamma node's exit
 * variables to the respective origin of the invariant variable.
 * x = ...
 * xo = gamma p xi
 * [xa <= xi]
 * [xr <= xa]
 * [xa <= xi]
 * [xr <= xa]
 * ... = anyOp xo
 * =>
 * x = ...
 * xo = gamma p xi
 * [xa <= xi]
 * [xo <= xa]
 * [xa <= xi]
 * [xo <= xa]
 * ... = anyOp x //xo changed to x
 *
 * @param node A gamma node that is supposed to be reduced.
 * @return True, if the transformation was successful, otherwise false.
 */
bool
ReduceGammaInvariantVariables(Node & node);

}

#endif

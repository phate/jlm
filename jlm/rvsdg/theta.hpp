/*
 * Copyright 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_THETA_HPP
#define JLM_RVSDG_THETA_HPP

#include <optional>

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/util/HashSet.hpp>

namespace jlm::rvsdg
{

class ThetaOperation final : public structural_op
{
public:
  ~ThetaOperation() noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;
};

class ThetaInput;
class ThetaOutput;
class ThetaResult;

class ThetaNode final : public structural_node
{
public:
  class loopvar_iterator
  {
  public:
    constexpr loopvar_iterator(ThetaOutput * output) noexcept
        : output_(output)
    {}

    const loopvar_iterator &
    operator++() noexcept;

    inline const loopvar_iterator
    operator++(int) noexcept
    {
      loopvar_iterator it(*this);
      ++(*this);
      return it;
    }

    inline bool
    operator==(const loopvar_iterator & other) const noexcept
    {
      return output_ == other.output_;
    }

    inline bool
    operator!=(const loopvar_iterator & other) const noexcept
    {
      return !(*this == other);
    }

    ThetaOutput *
    operator*() noexcept
    {
      return output_;
    }

    ThetaOutput **
    operator->() noexcept
    {
      return &output_;
    }

    ThetaOutput *
    output() const noexcept
    {
      return output_;
    }

  private:
    ThetaOutput * output_;
  };

  ~ThetaNode() noexcept override;

private:
  explicit ThetaNode(rvsdg::Region & parent);

public:
  /**
   * \brief Description of a loop-carried variable
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
    rvsdg::input * entry;
    /**
     * \brief Variable before iteration (input argument to subregion).
     */
    RegionArgument * pre;
    /**
     * \brief Variable after iteration (output result from subregion).
     */
    RegionResult * post;
    /**
     * \brief Variable at loop exit (output of theta).
     */
    rvsdg::output * exit;
  };

  static ThetaNode *
  create(rvsdg::Region * parent)
  {
    return new ThetaNode(*parent);
  }

  [[nodiscard]] rvsdg::Region *
  subregion() const noexcept
  {
    return structural_node::subregion(0);
  }

  [[nodiscard]] RegionResult *
  predicate() const noexcept
  {
    auto result = subregion()->result(0);
    JLM_ASSERT(dynamic_cast<const ctltype *>(&result->type()));
    return result;
  }

  inline void
  set_predicate(jlm::rvsdg::output * p)
  {
    auto node = node_output::node(predicate()->origin());

    predicate()->divert_to(p);
    if (node && !node->has_users())
      remove(node);
  }

  inline size_t
  nloopvars() const noexcept
  {
    JLM_ASSERT(ninputs() == noutputs());
    return ninputs();
  }

  inline ThetaNode::loopvar_iterator
  begin() const
  {
    if (ninputs() == 0)
      return loopvar_iterator(nullptr);

    return loopvar_iterator(output(0));
  }

  inline ThetaNode::loopvar_iterator
  end() const
  {
    return loopvar_iterator(nullptr);
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
    auto match = [](const ThetaOutput &)
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
  util::HashSet<const ThetaOutput *>
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
  util::HashSet<const ThetaOutput *>
  PruneThetaInputs()
  {
    auto match = [](const rvsdg::input &)
    {
      return true;
    };

    return RemoveThetaInputsWhere(match);
  }

  ThetaOutput *
  output(size_t index) const noexcept;

  ThetaOutput *
  add_loopvar(jlm::rvsdg::output * origin);

  LoopVar
  AddLoopVar(rvsdg::output * origin);

  virtual ThetaNode *
  copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const override;

  /**
   * \brief Map variable at entry to full varibale description
   *
   * \param input
   *   Input to the theta node
   *
   * \returns
   *   The loop variable description
   *
   * \pre
   *   \p input must be input to this node
   *
   * Returns the full description of the loop variable corresponding
   * to this entry into the theta node.
   */
  [[nodiscard]] LoopVar
  MapEntryLoopVar(const rvsdg::input & input) const;

  /**
   * \brief Map variable at entry to full varibale description
   *
   * \param argument
   *   Argument of theta region
   *
   * \returns
   *   The loop variable description
   *
   * \pre
   *   \p argument must be an argument to the subregion of this node
   *
   * Returns the full description of the loop variable corresponding
   * to this pre-loop value.
   */
  [[nodiscard]] LoopVar
  MapPreLoopVar(const RegionArgument & argument) const;

  /**
   * \brief Map variable at entry to full varibale description
   *
   * \param result
   *   Result of theta region
   *
   * \returns
   *   The loop variable description
   *
   * \pre
   *   \p result must be a result to the subregion of this node
   *
   * Returns the full description of the loop variable corresponding
   * to this post-loop value.
   */
  [[nodiscard]] LoopVar
  MapPostLoopVar(const RegionResult & result) const;

  /**
   * \brief Map variable at entry to full varibale description
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
  MapExitLoopVar(const rvsdg::output & output) const;

private:
  // Calling RemoveThetaInputsWhere/RemoveThetaOutputsWhere can result
  // in inputs (and pre-loop arguments) and outputs (and post-loop results)
  // become unmatched. In this case, the theta node itself has
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

class ThetaInput final : public structural_input
{
  friend ThetaNode;
  friend ThetaOutput;

public:
  ~ThetaInput() noexcept override;

  ThetaInput(ThetaNode * node, jlm::rvsdg::output * origin, std::shared_ptr<const rvsdg::type> type)
      : structural_input(node, origin, std::move(type)),
        output_(nullptr)
  {}

  ThetaNode *
  node() const noexcept
  {
    return static_cast<ThetaNode *>(structural_input::node());
  }

  ThetaOutput *
  output() const noexcept
  {
    return output_;
  }

  inline RegionArgument *
  argument() const noexcept
  {
    JLM_ASSERT(arguments.size() == 1);
    return arguments.first();
  }

  [[nodiscard]] inline RegionResult *
  result() const noexcept;

private:
  ThetaOutput * output_;
};

static inline bool
is_invariant(const ThetaInput * input) noexcept
{
  return input->result()->origin() == input->argument();
}

class ThetaOutput final : public structural_output
{
  friend ThetaNode;
  friend ThetaInput;

public:
  ~ThetaOutput() noexcept override;

  ThetaOutput(ThetaNode * node, const std::shared_ptr<const rvsdg::type> type)
      : structural_output(node, std::move(type)),
        input_(nullptr)
  {}

  ThetaNode *
  node() const noexcept
  {
    return static_cast<ThetaNode *>(structural_output::node());
  }

  [[nodiscard]] ThetaInput *
  input() const noexcept
  {
    return input_;
  }

  inline RegionArgument *
  argument() const noexcept
  {
    return input_->argument();
  }

  [[nodiscard]] RegionResult *
  result() const noexcept
  {
    JLM_ASSERT(results.size() == 1);
    return results.first();
  }

private:
  ThetaInput * input_;
};

/**
 * Represents a region argument in a theta subregion.
 */
class ThetaArgument final : public RegionArgument
{
  friend ThetaNode;

public:
  ~ThetaArgument() noexcept override;

  ThetaArgument &
  Copy(rvsdg::Region & region, structural_input * input) override;

private:
  ThetaArgument(rvsdg::Region & region, ThetaInput & input)
      : RegionArgument(&region, &input, input.Type())
  {
    JLM_ASSERT(is<ThetaOperation>(region.node()));
  }

  static ThetaArgument &
  Create(rvsdg::Region & region, ThetaInput & input)
  {
    auto thetaArgument = new ThetaArgument(region, input);
    region.append_argument(thetaArgument);
    return *thetaArgument;
  }
};

/**
 * Represents a region result in a theta subregion.
 */
class ThetaResult final : public RegionResult
{
  friend ThetaNode;

public:
  ~ThetaResult() noexcept override;

  ThetaResult &
  Copy(rvsdg::output & origin, jlm::rvsdg::structural_output * output) override;

private:
  ThetaResult(rvsdg::output & origin, ThetaOutput & thetaOutput)
      : RegionResult(origin.region(), &origin, &thetaOutput, origin.Type())
  {
    JLM_ASSERT(is<ThetaOperation>(origin.region()->node()));
  }

  static ThetaResult &
  Create(rvsdg::output & origin, ThetaOutput & thetaOutput)
  {
    auto thetaResult = new ThetaResult(origin, thetaOutput);
    origin.region()->append_result(thetaResult);
    return *thetaResult;
  }
};

/**
 * Represents the predicate result of a theta subregion.
 */
class ThetaPredicateResult final : public RegionResult
{
  friend ThetaNode;

public:
  ~ThetaPredicateResult() noexcept override;

  ThetaPredicateResult &
  Copy(rvsdg::output & origin, structural_output * output) override;

private:
  explicit ThetaPredicateResult(rvsdg::output & origin)
      : RegionResult(origin.region(), &origin, nullptr, ctltype::Create(2))
  {
    JLM_ASSERT(is<ThetaOperation>(origin.region()->node()));
  }

  static ThetaPredicateResult &
  Create(rvsdg::output & origin)
  {
    auto thetaResult = new ThetaPredicateResult(origin);
    origin.region()->append_result(thetaResult);
    return *thetaResult;
  }
};

static inline bool
is_invariant(const ThetaOutput * output) noexcept
{
  return output->result()->origin() == output->argument();
}

/* theta node method definitions */

inline ThetaOutput *
ThetaNode::output(size_t index) const noexcept
{
  return static_cast<ThetaOutput *>(node::output(index));
}

template<typename F>
util::HashSet<const rvsdg::input *>
ThetaNode::RemoveThetaOutputsWhere(const F & match)
{
  util::HashSet<const rvsdg::input *> deadInputs;

  // iterate backwards to avoid the invalidation of 'n' by RemoveOutput()
  for (size_t n = noutputs() - 1; n != static_cast<size_t>(-1); n--)
  {
    auto & thetaOutput = *output(n);
    auto & thetaResult = *thetaOutput.result();

    if (thetaOutput.IsDead() && match(thetaOutput))
    {
      deadInputs.Insert(thetaOutput.input());
      subregion()->RemoveResult(thetaResult.index());
      MarkOutputIndexErased(thetaOutput.index());
      RemoveOutput(thetaOutput.index());
    }
  }

  return deadInputs;
}

template<typename F>
util::HashSet<const ThetaOutput *>
ThetaNode::RemoveThetaInputsWhere(const F & match)
{
  util::HashSet<const ThetaOutput *> deadOutputs;

  // iterate backwards to avoid the invalidation of 'n' by RemoveInput()
  for (size_t n = ninputs() - 1; n != static_cast<size_t>(-1); n--)
  {
    auto & thetaInput = *input(n);
    auto loopvar = MapEntryLoopVar(thetaInput);

    if (loopvar.pre->IsDead() && match(thetaInput))
    {
      deadOutputs.Insert(jlm::util::AssertedCast<ThetaOutput>(loopvar.exit));
      subregion()->RemoveArgument(loopvar.pre->index());
      MarkInputIndexErased(thetaInput.index());
      RemoveInput(thetaInput.index());
    }
  }

  return deadOutputs;
}

/* theta input method definitions */

[[nodiscard]] inline RegionResult *
ThetaInput::result() const noexcept
{
  return output_->result();
}

}

#endif

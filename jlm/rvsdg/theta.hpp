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
  util::HashSet<const ThetaInput *>
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
  util::HashSet<const ThetaInput *>
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
   * @tparam F A type that supports the function call operator: bool operator(const ThetaInput&)
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
    auto match = [](const ThetaInput &)
    {
      return true;
    };

    return RemoveThetaInputsWhere(match);
  }

  ThetaInput *
  input(size_t index) const noexcept;

  ThetaOutput *
  output(size_t index) const noexcept;

  ThetaOutput *
  add_loopvar(jlm::rvsdg::output * origin);

  virtual ThetaNode *
  copy(rvsdg::Region * region, jlm::rvsdg::substitution_map & smap) const override;
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

inline ThetaInput *
ThetaNode::input(size_t index) const noexcept
{
  return static_cast<ThetaInput *>(node::input(index));
}

inline ThetaOutput *
ThetaNode::output(size_t index) const noexcept
{
  return static_cast<ThetaOutput *>(node::output(index));
}

template<typename F>
util::HashSet<const ThetaInput *>
ThetaNode::RemoveThetaOutputsWhere(const F & match)
{
  util::HashSet<const ThetaInput *> deadInputs;

  // iterate backwards to avoid the invalidation of 'n' by RemoveOutput()
  for (size_t n = noutputs() - 1; n != static_cast<size_t>(-1); n--)
  {
    auto & thetaOutput = *output(n);
    auto & thetaResult = *thetaOutput.result();

    if (thetaOutput.IsDead() && match(thetaOutput))
    {
      deadInputs.Insert(thetaOutput.input());
      subregion()->RemoveResult(thetaResult.index());
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
    auto & thetaArgument = *thetaInput.argument();

    if (thetaArgument.IsDead() && match(thetaInput))
    {
      deadOutputs.Insert(thetaInput.output());
      subregion()->RemoveArgument(thetaArgument.index());
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

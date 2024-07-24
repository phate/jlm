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

/* theta operation */

class theta_op final : public structural_op
{
public:
  virtual ~theta_op() noexcept;
  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;
};

/* theta node */

class theta_input;
class theta_output;

class theta_node final : public structural_node
{
public:
  class loopvar_iterator
  {
  public:
    inline constexpr loopvar_iterator(jlm::rvsdg::theta_output * output) noexcept
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

    inline theta_output *
    operator*() noexcept
    {
      return output_;
    }

    inline theta_output **
    operator->() noexcept
    {
      return &output_;
    }

    inline jlm::rvsdg::theta_output *
    output() const noexcept
    {
      return output_;
    }

  private:
    jlm::rvsdg::theta_output * output_;
  };

  virtual ~theta_node();

private:
  inline theta_node(jlm::rvsdg::region * parent)
      : structural_node(jlm::rvsdg::theta_op(), parent, 1)
  {
    auto predicate = jlm::rvsdg::control_false(subregion());
    result::create(subregion(), predicate, nullptr, ctltype::Create(2));
  }

public:
  static jlm::rvsdg::theta_node *
  create(jlm::rvsdg::region * parent)
  {
    return new jlm::rvsdg::theta_node(parent);
  }

  inline jlm::rvsdg::region *
  subregion() const noexcept
  {
    return structural_node::subregion(0);
  }

  inline jlm::rvsdg::result *
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

  inline theta_node::loopvar_iterator
  begin() const
  {
    if (ninputs() == 0)
      return loopvar_iterator(nullptr);

    return loopvar_iterator(output(0));
  }

  inline theta_node::loopvar_iterator
  end() const
  {
    return loopvar_iterator(nullptr);
  }

  /**
   * Remove theta outputs and their respective results.
   *
   * An output must match the condition specified by \p match and it must be dead.
   *
   * @tparam F A type that supports the function call operator: bool operator(const theta_output&)
   * @param match Defines the condition of the elements to remove.
   * @return The inputs corresponding to the removed outputs.
   *
   * \note The application of this method might leave the theta node in an invalid state. Some
   * inputs might refer to outputs that have been removed by the application of this method. It
   * is up to the caller to ensure that the invariants of the theta node will eventually be met
   * again.
   *
   * \see RemoveThetaInputsWhere()
   * \see theta_output#IsDead()
   */
  template<typename F>
  util::HashSet<const theta_input *>
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
   * \see theta_output#IsDead()
   */
  util::HashSet<const theta_input *>
  PruneThetaOutputs()
  {
    auto match = [](const theta_output &)
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
   * @tparam F A type that supports the function call operator: bool operator(const theta_input&)
   * @param match Defines the condition of the elements to remove.
   * @return The outputs corresponding to the removed outputs.
   *
   * \note The application of this method might leave the theta node in an invalid state. Some
   * outputs might refer to inputs that have been removed by the application of this method. It
   * is up to the caller to ensure that the invariants of the theta node will eventually be met
   * again.
   *
   * \see RemoveThetaOutputsWhere()
   * \see argument#IsDead()
   */
  template<typename F>
  util::HashSet<const theta_output *>
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
   * \see argument#IsDead()
   */
  util::HashSet<const theta_output *>
  PruneThetaInputs()
  {
    auto match = [](const theta_input &)
    {
      return true;
    };

    return RemoveThetaInputsWhere(match);
  }

  theta_input *
  input(size_t index) const noexcept;

  theta_output *
  output(size_t index) const noexcept;

  jlm::rvsdg::theta_output *
  add_loopvar(jlm::rvsdg::output * origin);

  virtual jlm::rvsdg::theta_node *
  copy(jlm::rvsdg::region * region, jlm::rvsdg::substitution_map & smap) const override;
};

/* theta input */

class theta_input final : public structural_input
{
  friend theta_node;
  friend theta_output;

public:
  virtual ~theta_input() noexcept;

  inline theta_input(
      theta_node * node,
      jlm::rvsdg::output * origin,
      std::shared_ptr<const rvsdg::type> type)
      : structural_input(node, origin, std::move(type)),
        output_(nullptr)
  {}

  theta_node *
  node() const noexcept
  {
    return static_cast<theta_node *>(structural_input::node());
  }

  inline jlm::rvsdg::theta_output *
  output() const noexcept
  {
    return output_;
  }

  inline jlm::rvsdg::argument *
  argument() const noexcept
  {
    JLM_ASSERT(arguments.size() == 1);
    return arguments.first();
  }

  jlm::rvsdg::result *
  result() const noexcept;

private:
  jlm::rvsdg::theta_output * output_;
};

static inline bool
is_theta_input(const jlm::rvsdg::input * input) noexcept
{
  return dynamic_cast<const jlm::rvsdg::theta_input *>(input) != nullptr;
}

static inline bool
is_invariant(const jlm::rvsdg::theta_input * input) noexcept
{
  return input->result()->origin() == input->argument();
}

/* theta output */

class theta_output final : public structural_output
{
  friend theta_node;
  friend theta_input;

public:
  virtual ~theta_output() noexcept;

  inline theta_output(theta_node * node, const std::shared_ptr<const rvsdg::type> type)
      : structural_output(node, std::move(type)),
        input_(nullptr)
  {}

  theta_node *
  node() const noexcept
  {
    return static_cast<theta_node *>(structural_output::node());
  }

  inline jlm::rvsdg::theta_input *
  input() const noexcept
  {
    return input_;
  }

  inline jlm::rvsdg::argument *
  argument() const noexcept
  {
    return input_->argument();
  }

  inline jlm::rvsdg::result *
  result() const noexcept
  {
    JLM_ASSERT(results.size() == 1);
    return results.first();
  }

private:
  jlm::rvsdg::theta_input * input_;
};

/**
 * Represents a region argument in a theta subregion.
 */
class ThetaArgument final : public argument
{
  friend theta_node;

public:
  ~ThetaArgument() noexcept override;

private:
  ThetaArgument(rvsdg::region & region, theta_input & input)
      : argument(&region, &input, input.Type())
  {
    JLM_ASSERT(is<theta_op>(region.node()));
  }

  static ThetaArgument &
  Create(rvsdg::region & region, theta_input & input)
  {
    auto thetaArgument = new ThetaArgument(region, input);
    region.append_argument(thetaArgument);
    return *thetaArgument;
  }
};

/**
 * Represents a region result in a theta subregion.
 */
class ThetaResult final : public result
{
  friend theta_node;

public:
  ~ThetaResult() noexcept override;

private:
  ThetaResult(ThetaArgument & thetaArgument, theta_output & thetaOutput)
      : result(thetaArgument.region(), &thetaArgument, &thetaOutput, thetaArgument.Type())
  {
    JLM_ASSERT(is<theta_op>(thetaArgument.region()->node()));
  }

  static ThetaResult &
  Create(ThetaArgument & thetaArgument, theta_output & thetaOutput)
  {
    auto thetaResult = new ThetaResult(thetaArgument, thetaOutput);
    thetaArgument.region()->append_result(thetaResult);
    return *thetaResult;
  }
};

static inline bool
is_invariant(const jlm::rvsdg::theta_output * output) noexcept
{
  return output->result()->origin() == output->argument();
}

/* theta node method definitions */

inline jlm::rvsdg::theta_input *
theta_node::input(size_t index) const noexcept
{
  return static_cast<theta_input *>(node::input(index));
}

inline jlm::rvsdg::theta_output *
theta_node::output(size_t index) const noexcept
{
  return static_cast<theta_output *>(node::output(index));
}

template<typename F>
util::HashSet<const theta_input *>
theta_node::RemoveThetaOutputsWhere(const F & match)
{
  util::HashSet<const theta_input *> deadInputs;

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
util::HashSet<const theta_output *>
theta_node::RemoveThetaInputsWhere(const F & match)
{
  util::HashSet<const theta_output *> deadOutputs;

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

inline jlm::rvsdg::result *
theta_input::result() const noexcept
{
  return output_->result();
}

}

#endif

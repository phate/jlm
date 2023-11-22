/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_IR_HLS_HPP
#define JLM_HLS_IR_HLS_HPP

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/operation.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/util/common.hpp>

namespace jlm::hls
{

class branch_op final : public jlm::rvsdg::simple_op
{
private:
  branch_op(size_t nalternatives, const jlm::rvsdg::type & type, bool loop)
      : jlm::rvsdg::simple_op(
          { jlm::rvsdg::ctltype(nalternatives), type },
          std::vector<jlm::rvsdg::port>(nalternatives, type)),
        loop(loop)
  {}

public:
  virtual ~branch_op()
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const branch_op *>(&other);
    // check predicate and value
    return ot && ot->argument(0).type() == argument(0).type()
        && ot->result(0).type() == result(0).type();
  }

  std::string
  debug_string() const override
  {
    return "HLS_BRANCH";
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new branch_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & predicate, jlm::rvsdg::output & value, bool loop = false)
  {
    auto ctl = dynamic_cast<const jlm::rvsdg::ctltype *>(&predicate.type());
    if (!ctl)
      throw util::error("Predicate needs to be a ctltype.");

    auto region = predicate.region();
    branch_op op(ctl->nalternatives(), value.type(), loop);
    return jlm::rvsdg::simple_node::create_normalized(region, op, { &predicate, &value });
  }

  bool loop; // only used for dot output
};

class fork_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~fork_op()
  {}

  fork_op(size_t nalternatives, const jlm::rvsdg::type & type)
      : jlm::rvsdg::simple_op({ type }, std::vector<jlm::rvsdg::port>(nalternatives, type))
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const fork_op *>(&other);
    // check predicate and value
    return ot && ot->argument(0).type() == argument(0).type() && ot->nresults() == nresults();
  }

  std::string
  debug_string() const override
  {
    return "HLS_FORK";
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new fork_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(size_t nalternatives, jlm::rvsdg::output & value)
  {

    auto region = value.region();
    fork_op op(nalternatives, value.type());
    return jlm::rvsdg::simple_node::create_normalized(region, op, { &value });
  }
};

class merge_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~merge_op()
  {}

  merge_op(size_t nalternatives, const jlm::rvsdg::type & type)
      : jlm::rvsdg::simple_op(std::vector<jlm::rvsdg::port>(nalternatives, type), { type })
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const merge_op *>(&other);
    return ot && ot->narguments() == narguments() && ot->argument(0).type() == argument(0).type();
  }

  std::string
  debug_string() const override
  {
    return "HLS_MERGE";
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new merge_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(const std::vector<jlm::rvsdg::output *> & alternatives)
  {
    if (alternatives.empty())
      throw util::error("Insufficient number of operands.");

    auto region = alternatives.front()->region();
    merge_op op(alternatives.size(), alternatives.front()->type());
    return jlm::rvsdg::simple_node::create_normalized(region, op, alternatives);
  }
};

class mux_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~mux_op()
  {}

  mux_op(size_t nalternatives, const jlm::rvsdg::type & type, bool discarding, bool loop)
      : jlm::rvsdg::simple_op(create_portvector(nalternatives, type), { type }),
        discarding(discarding),
        loop(loop)
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const mux_op *>(&other);
    // check predicate and value
    return ot && ot->argument(0).type() == argument(0).type()
        && ot->result(0).type() == result(0).type() && ot->discarding == discarding;
  }

  std::string
  debug_string() const override
  {
    return discarding ? "HLS_DMUX" : "HLS_NDMUX";
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new mux_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(
      jlm::rvsdg::output & predicate,
      const std::vector<jlm::rvsdg::output *> & alternatives,
      bool discarding,
      bool loop = false)
  {
    if (alternatives.empty())
      throw util::error("Insufficient number of operands.");
    auto ctl = dynamic_cast<const jlm::rvsdg::ctltype *>(&predicate.type());
    if (!ctl)
      throw util::error("Predicate needs to be a ctltype.");
    if (alternatives.size() != ctl->nalternatives())
      throw util::error("Alternatives and predicate do not match.");

    auto region = predicate.region();
    auto operands = std::vector<jlm::rvsdg::output *>();
    operands.push_back(&predicate);
    operands.insert(operands.end(), alternatives.begin(), alternatives.end());
    mux_op op(alternatives.size(), alternatives.front()->type(), discarding, loop);
    return jlm::rvsdg::simple_node::create_normalized(region, op, operands);
  }

  bool discarding;
  bool loop; // used only for dot output
private:
  static std::vector<jlm::rvsdg::port>
  create_portvector(size_t nalternatives, const jlm::rvsdg::type & type)
  {
    auto vec = std::vector<jlm::rvsdg::port>(nalternatives + 1, type);
    vec[0] = jlm::rvsdg::ctltype(nalternatives);
    return vec;
  }
};

class sink_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~sink_op()
  {}

  sink_op(const jlm::rvsdg::type & type)
      : jlm::rvsdg::simple_op({ type }, {})
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const sink_op *>(&other);
    return ot && ot->argument(0).type() == argument(0).type();
  }

  std::string
  debug_string() const override
  {
    return "HLS_SINK";
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new sink_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & value)
  {
    auto region = value.region();
    sink_op op(value.type());
    return jlm::rvsdg::simple_node::create_normalized(region, op, { &value });
  }
};

class predicate_buffer_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~predicate_buffer_op()
  {}

  predicate_buffer_op(const jlm::rvsdg::ctltype & type)
      : jlm::rvsdg::simple_op({ type }, { type })
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const predicate_buffer_op *>(&other);
    return ot && ot->result(0).type() == result(0).type();
  }

  std::string
  debug_string() const override
  {
    return "HLS_PRED_BUF";
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new predicate_buffer_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & predicate)
  {
    auto region = predicate.region();
    auto ctl = dynamic_cast<const jlm::rvsdg::ctltype *>(&predicate.type());
    if (!ctl)
      throw util::error("Predicate needs to be a ctltype.");
    predicate_buffer_op op(*ctl);
    return jlm::rvsdg::simple_node::create_normalized(region, op, { &predicate });
  }
};

class buffer_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~buffer_op()
  {}

  buffer_op(const jlm::rvsdg::type & type, size_t capacity, bool pass_through)
      : jlm::rvsdg::simple_op({ type }, { type }),
        capacity(capacity),
        pass_through(pass_through)
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const buffer_op *>(&other);
    return ot && ot->capacity == capacity && ot->pass_through == pass_through
        && ot->result(0).type() == result(0).type();
  }

  std::string
  debug_string() const override
  {
    return util::strfmt("HLS_BUF_", (pass_through ? "P_" : ""), capacity);
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new buffer_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & value, size_t capacity, bool pass_through = false)
  {
    auto region = value.region();
    buffer_op op(value.type(), capacity, pass_through);
    return jlm::rvsdg::simple_node::create_normalized(region, op, { &value });
  }

  size_t capacity;
  bool pass_through;

private:
};

class triggertype final : public jlm::rvsdg::statetype
{
public:
  virtual ~triggertype()
  {}

  triggertype()
      : jlm::rvsdg::statetype()
  {}

  std::string
  debug_string() const override
  {
    return "trigger";
  };

  bool
  operator==(const jlm::rvsdg::type & other) const noexcept override
  {
    auto type = dynamic_cast<const triggertype *>(&other);
    return type;
  };

  virtual std::unique_ptr<jlm::rvsdg::type>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::type>(new triggertype(*this));
  }

private:
};

const triggertype trigger;

class trigger_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~trigger_op()
  {}

  trigger_op(const jlm::rvsdg::type & type)
      : jlm::rvsdg::simple_op({ trigger, type }, { type })
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const trigger_op *>(&other);
    // check predicate and value
    return ot && ot->argument(1).type() == argument(1).type()
        && ot->result(0).type() == result(0).type();
  }

  std::string
  debug_string() const override
  {
    return "HLS_TRIGGER";
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new trigger_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & tg, jlm::rvsdg::output & value)
  {
    if (tg.type() != trigger)
      throw util::error("Trigger needs to be a triggertype.");

    auto region = value.region();
    trigger_op op(value.type());
    return jlm::rvsdg::simple_node::create_normalized(region, op, { &tg, &value });
  }
};

class print_op final : public jlm::rvsdg::simple_op
{
private:
  size_t _id;

public:
  virtual ~print_op()
  {}

  print_op(const jlm::rvsdg::type & type)
      : jlm::rvsdg::simple_op({ type }, { type })
  {
    static size_t common_id{ 0 };
    _id = common_id++;
  }

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    //				auto ot = dynamic_cast<const print_op *>(&other);
    // check predicate and value
    //				return ot
    //					   && ot->argument(0).type() == argument(0).type()
    //					   && ot->result(0).type() == result(0).type();
    return false; // print nodes are intentionally distinct
  }

  std::string
  debug_string() const override
  {
    return util::strfmt("HLS_PRINT_", _id);
  }

  size_t
  id() const
  {
    return _id;
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new print_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & value)
  {

    auto region = value.region();
    print_op op(value.type());
    return jlm::rvsdg::simple_node::create_normalized(region, op, { &value });
  }
};

class loop_op final : public jlm::rvsdg::structural_op
{
public:
  virtual ~loop_op() noexcept
  {}

  std::string
  debug_string() const override
  {
    return "HLS_LOOP";
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new loop_op(*this));
  }
};

class backedge_argument;
class backedge_result;
class loop_node;

class backedge_argument : public jlm::rvsdg::argument
{
  friend loop_node;
  friend backedge_result;

public:
  ~backedge_argument() override = default;

  backedge_result *
  result()
  {
    return result_;
  }

private:
  backedge_argument(jlm::rvsdg::region * region, const jlm::rvsdg::type & type)
      : jlm::rvsdg::argument(region, nullptr, type),
        result_(nullptr)
  {}

  static backedge_argument *
  create(jlm::rvsdg::region * region, const jlm::rvsdg::type & type)
  {
    auto argument = new backedge_argument(region, type);
    region->append_argument(argument);
    return argument;
  }

  backedge_result * result_;
};

class backedge_result : public jlm::rvsdg::result
{
  friend loop_node;
  friend backedge_argument;

public:
  ~backedge_result() override = default;

  backedge_argument *
  argument()
  {
    return argument_;
  }

private:
  backedge_result(jlm::rvsdg::output * origin)
      : jlm::rvsdg::result(origin->region(), origin, nullptr, origin->port()),
        argument_(nullptr)
  {}

  static backedge_result *
  create(jlm::rvsdg::output * origin)
  {
    auto result = new backedge_result(origin);
    origin->region()->append_result(result);
    return result;
  }

  backedge_argument * argument_;
};

class loop_node final : public jlm::rvsdg::structural_node
{
public:
  virtual ~loop_node()
  {}

private:
  inline loop_node(jlm::rvsdg::region * parent)
      : structural_node(loop_op(), parent, 1)
  {}

  jlm::rvsdg::output * _predicate_buffer;

public:
  static loop_node *
  create(jlm::rvsdg::region * parent, bool init = true);

  inline jlm::rvsdg::region *
  subregion() const noexcept
  {
    return structural_node::subregion(0);
  }

  inline jlm::rvsdg::result *
  predicate() const noexcept
  {
    auto result = subregion()->result(0);
    JLM_ASSERT(dynamic_cast<const jlm::rvsdg::ctltype *>(&result->type()));
    return result;
  }

  inline jlm::rvsdg::output *
  predicate_buffer() const noexcept
  {
    return _predicate_buffer;
  }

  void
  set_predicate(jlm::rvsdg::output * p);

  backedge_argument *
  add_backedge(const jlm::rvsdg::type & type);

  jlm::rvsdg::structural_output *
  add_loopvar(jlm::rvsdg::output * origin, jlm::rvsdg::output ** buffer = nullptr);

  virtual loop_node *
  copy(jlm::rvsdg::region * region, jlm::rvsdg::substitution_map & smap) const override;

  /**
   * Remove loop inputs and their respective arguments.
   *
   * An input must match the condition specified by \p match and its respective argument must be
   * dead.
   *
   * @tparam F A type that supports the function call operator:
   * bool operator(const structural_input&)
   * @param match Defines the condition of the elements to remove.
   * @return The number of removed inputs.
   *
   * \note The application of this method might leave the loop node in an invalid state. Some
   * outputs might refer to inputs that have been removed by the application of this method. It
   * is up to the caller to ensure that the invariants of the loop node will eventually be met
   * again.
   *
   * \see argument#IsDead()
   */
  template<typename F>
  size_t
  RemoveLoopInputsWhere(const F & match)
  {
    size_t numRemovedInputs = 0;

    // iterate backwards to avoid the invalidation of 'n' by RemoveInput()
    for (size_t n = ninputs() - 1; n != static_cast<size_t>(-1); n--)
    {
      auto & loopInput = *input(n);
      JLM_ASSERT(loopInput.arguments.size() == 1);
      auto & loopArgument = *loopInput.arguments.begin();

      if (loopArgument.IsDead() && match(loopInput))
      {
        subregion()->RemoveArgument(loopArgument.index());
        RemoveInput(loopInput.index());
        numRemovedInputs++;
      }
    }

    return numRemovedInputs;
  }

  /**
   * Remove all dead loop inputs and their respective arguments.
   *
   * @return The number of removed inputs.
   *
   * \note The application of this method might leave the loop node in an invalid state. It
   * is up to the caller to ensure that the invariants of the loop node will eventually be met
   * again.
   *
   * \see RemoveLoopInputsWhere()
   * \see argument#IsDead()
   */
  size_t
  PruneLoopInputs()
  {
    auto match = [](const rvsdg::structural_input &)
    {
      return true;
    };

    return RemoveLoopInputsWhere(match);
  }

  /**
   * Remove loop outputs and their respective results.
   *
   * An output must match the condition specified by \p match and it must be dead.
   *
   * @tparam F A type that supports the function call operator:
   * bool operator(const structural_output&)
   * @param match Defines the condition of the elements to remove.
   * @return The number of removed outputs.
   *
   * \note The application of this method might leave the loop node in an invalid state. Some
   * inputs might refer to outputs that have been removed by the application of this method. It
   * is up to the caller to ensure that the invariants of the loop node will eventually be met
   * again.
   *
   * \see output#IsDead()
   */
  template<typename F>
  size_t
  RemoveLoopOutputsWhere(const F & match)
  {
    size_t numRemovedOutputs = 0;

    // iterate backwards to avoid the invalidation of 'n' by RemoveOutput()
    for (size_t n = noutputs() - 1; n != static_cast<size_t>(-1); n--)
    {
      auto & loopOutput = *output(n);
      JLM_ASSERT(loopOutput.results.size() == 1);
      auto & loopResult = *loopOutput.results.begin();

      if (loopOutput.IsDead() && match(loopOutput))
      {
        subregion()->RemoveResult(loopResult.index());
        RemoveOutput(loopOutput.index());
        numRemovedOutputs++;
      }
    }

    return numRemovedOutputs;
  }

  /**
   * Remove all dead loop outputs and their respective results.
   *
   * @return The number of removed outputs.
   *
   * \note The application of this method might leave the loop node in an invalid state. It
   * is up to the caller to ensure that the invariants of the loop node will eventually be met
   * again.
   *
   * \see RemoveLoopOutputsWhere()
   * \see output#IsDead()
   */
  size_t
  PruneLoopOutputs()
  {
    auto match = [](const rvsdg::structural_output &)
    {
      return true;
    };

    return RemoveLoopOutputsWhere(match);
  }

  /**
   * Remove back-edge arguments and their respective results.
   *
   * An argument must match the condition specified by \p match.
   *
   * @tparam F A type that supports the function call operator:
   * bool operator(const backedge_argument&)
   * @param match Defines the condition of the elements to remove.
   * @return The number of removed arguments.
   *
   * \note The application of this method might leave the loop node in an invalid state. It
   * is up to the caller to ensure that the invariants of the loop node will eventually be met
   * again.
   *
   */
  template<typename F>
  size_t
  RemoveBackEdgeArgumentsWhere(const F & match)
  {
    size_t numRemovedArguments = 0;
    auto & subregion = *this->subregion();

    // iterate backwards to avoid the invalidation of 'n' by RemoveArgument()
    for (size_t n = subregion.narguments() - 1; n != static_cast<size_t>(-1); n--)
    {
      auto backEdgeArgument = dynamic_cast<backedge_argument *>(subregion.argument(n));
      if (backEdgeArgument && match(*backEdgeArgument))
      {
        auto & backEdgeResult = *backEdgeArgument->result();

        subregion.RemoveResult(backEdgeResult.index());
        subregion.RemoveArgument(backEdgeArgument->index());
        numRemovedArguments++;
      }
    }

    return numRemovedArguments;
  }
};

}

#endif // JLM_HLS_IR_HLS_HPP

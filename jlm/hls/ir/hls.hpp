/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_IR_HLS_HPP
#define JLM_HLS_IR_HLS_HPP

#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/operation.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/util/common.hpp>

#include <memory>
#include <utility>

namespace jlm::hls
{
/**
 * @return The size of a pointer in bits.
 */
[[nodiscard]] size_t
GetPointerSizeInBits();

int
JlmSize(const jlm::rvsdg::Type * type);

class branch_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~branch_op()
  {}

  branch_op(size_t nalternatives, const std::shared_ptr<const jlm::rvsdg::Type> & type, bool loop)
      : SimpleOperation(
            { rvsdg::ControlType::Create(nalternatives), type },
            { nalternatives, type }),
        loop(loop)
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const branch_op *>(&other);
    // check predicate and value
    return ot && ot->loop == loop && *ot->argument(0) == *argument(0)
        && *ot->result(0) == *result(0);
  }

  std::string
  debug_string() const override
  {
    return "HLS_BRANCH";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<branch_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & predicate, jlm::rvsdg::output & value, bool loop = false)
  {
    auto ctl = dynamic_cast<const rvsdg::ControlType *>(&predicate.type());
    if (!ctl)
      throw util::error("Predicate needs to be a control type.");

    return outputs(&rvsdg::CreateOpNode<branch_op>(
        { &predicate, &value },
        ctl->nalternatives(),
        value.Type(),
        loop));
  }

  bool loop; // only used for dot output
};

/**
 * Forks ensures 1-to-1 connections between producers and consumers, i.e., they handle fanout of
 * signals. Normal forks have a register inside to ensure that a token consumed on one output is not
 * repeated. The fork only creates an acknowledge on its single input once all outputs have been
 * consumed.
 *
 * CFORK (constant fork):
 * Handles the special case when the same constant is used as input for multiple nodes. It would be
 * possible to have a constant for each input, but deduplication replaces the constants with a
 * single constant fork. Since the input of the fork is always the same value and is always valid.
 * No handshaking is necessary and the outputs of the fork is always valid.
 */
class fork_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~fork_op()
  {}

  /**
   * Create a fork operation that is not a constant fork.
   *
   * /param nalternatives Number of outputs.
   * /param value The signal type, which is the same for the input and all outputs.
   */
  fork_op(size_t nalternatives, const std::shared_ptr<const jlm::rvsdg::Type> & type)
      : SimpleOperation({ type }, { nalternatives, type })
  {}

  /**
   * Create a fork operation.
   *
   * /param nalternatives Number of outputs.
   * /param value The signal type, which is the same for the input and all outputs.
   * /param isConstant If true, the fork is a constant fork.
   */
  fork_op(
      size_t nalternatives,
      const std::shared_ptr<const jlm::rvsdg::Type> & type,
      bool isConstant)
      : SimpleOperation({ type }, { nalternatives, type }),
        IsConstant_(isConstant)
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    auto forkOp = dynamic_cast<const fork_op *>(&other);
    // check predicate and value
    return forkOp && *forkOp->argument(0) == *argument(0) && forkOp->nresults() == nresults()
        && forkOp->IsConstant() == IsConstant_;
  }

  /**
   * Debug string for the fork operation.
   * /return HLS_CFORK if the fork is a constant fork, else HLS_FORK.
   */
  std::string
  debug_string() const override
  {
    return IsConstant() ? "HLS_CFORK" : "HLS_FORK";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<fork_op>(*this);
  }

  /**
   * Create a fork operation with a single input and multiple outputs.
   *
   * /param nalternatives Number of outputs.
   * /param value The signal type, which is the same for the input and all outputs.
   * /param isConstant If true, the fork is a constant fork.
   *
   * /return A vector of outputs.
   */
  static std::vector<jlm::rvsdg::output *>
  create(size_t nalternatives, jlm::rvsdg::output & value, bool isConstant = false)
  {
    return outputs(
        &rvsdg::CreateOpNode<fork_op>({ &value }, nalternatives, value.Type(), isConstant));
  }

  /**
   * Check if a fork is a constant fork (CFORK).
   *
   * /return True if the fork is a constant fork, i.e., the input of the fork is a constant, else
   * false.
   */
  [[nodiscard]] bool
  IsConstant() const noexcept
  {
    return IsConstant_;
  }

private:
  bool IsConstant_ = false;
};

class merge_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~merge_op()
  {}

  merge_op(size_t nalternatives, const std::shared_ptr<const jlm::rvsdg::Type> & type)
      : SimpleOperation({ nalternatives, type }, { type })
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const merge_op *>(&other);
    return ot && ot->narguments() == narguments() && *ot->argument(0) == *argument(0);
  }

  std::string
  debug_string() const override
  {
    return "HLS_MERGE";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<merge_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(const std::vector<jlm::rvsdg::output *> & alternatives)
  {
    if (alternatives.empty())
      throw util::error("Insufficient number of operands.");

    return outputs(&rvsdg::CreateOpNode<merge_op>(
        *alternatives.front()->region(),
        alternatives.size(),
        alternatives.front()->Type()));
  }
};

class mux_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~mux_op()
  {}

  mux_op(
      size_t nalternatives,
      const std::shared_ptr<const jlm::rvsdg::Type> & type,
      bool discarding,
      bool loop)
      : SimpleOperation(create_typevector(nalternatives, type), { type }),
        discarding(discarding),
        loop(loop)
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const mux_op *>(&other);
    // check predicate and value
    return ot && *ot->argument(0) == *argument(0) && *ot->result(0) == *result(0)
        && ot->discarding == discarding;
  }

  std::string
  debug_string() const override
  {
    return discarding ? "HLS_DMUX" : "HLS_NDMUX";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<mux_op>(*this);
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
    auto ctl = dynamic_cast<const rvsdg::ControlType *>(&predicate.type());
    if (!ctl)
      throw util::error("Predicate needs to be a control type.");
    if (alternatives.size() != ctl->nalternatives())
      throw util::error("Alternatives and predicate do not match.");

    auto operands = std::vector<jlm::rvsdg::output *>();
    operands.push_back(&predicate);
    operands.insert(operands.end(), alternatives.begin(), alternatives.end());
    return outputs(&rvsdg::CreateOpNode<mux_op>(
        operands,
        alternatives.size(),
        alternatives.front()->Type(),
        discarding,
        loop));
  }

  bool discarding;
  bool loop; // used only for dot output
private:
  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  create_typevector(size_t nalternatives, std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    auto vec =
        std::vector<std::shared_ptr<const jlm::rvsdg::Type>>(nalternatives + 1, std::move(type));
    vec[0] = rvsdg::ControlType::Create(nalternatives);
    return vec;
  }
};

class sink_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~sink_op()
  {}

  explicit sink_op(const std::shared_ptr<const jlm::rvsdg::Type> & type)
      : SimpleOperation({ type }, {})
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const sink_op *>(&other);
    return ot && *ot->argument(0) == *argument(0);
  }

  std::string
  debug_string() const override
  {
    return "HLS_SINK";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<sink_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & value)
  {
    return outputs(&rvsdg::CreateOpNode<sink_op>({ &value }, value.Type()));
  }
};

class predicate_buffer_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~predicate_buffer_op()
  {}

  explicit predicate_buffer_op(const std::shared_ptr<const rvsdg::ControlType> & type)
      : SimpleOperation({ type }, { type })
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const predicate_buffer_op *>(&other);
    return ot && *ot->result(0) == *result(0);
  }

  std::string
  debug_string() const override
  {
    return "HLS_PRED_BUF";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<predicate_buffer_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & predicate)
  {
    auto ctl = std::dynamic_pointer_cast<const rvsdg::ControlType>(predicate.Type());
    if (!ctl)
      throw util::error("Predicate needs to be a control type.");

    return outputs(&rvsdg::CreateOpNode<predicate_buffer_op>({ &predicate }, ctl));
  }
};

class loop_constant_buffer_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~loop_constant_buffer_op()
  {}

  loop_constant_buffer_op(
      const std::shared_ptr<const rvsdg::ControlType> & ctltype,
      const std::shared_ptr<const jlm::rvsdg::Type> & type)
      : SimpleOperation({ ctltype, type }, { type })
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const loop_constant_buffer_op *>(&other);
    return ot && *ot->result(0) == *result(0) && *ot->argument(0) == *argument(0);
  }

  std::string
  debug_string() const override
  {
    return "HLS_LOOP_CONST_BUF";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<loop_constant_buffer_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & predicate, jlm::rvsdg::output & value)
  {
    auto ctl = std::dynamic_pointer_cast<const rvsdg::ControlType>(predicate.Type());
    if (!ctl)
      throw util::error("Predicate needs to be a control type.");

    return outputs(
        &rvsdg::CreateOpNode<loop_constant_buffer_op>({ &predicate, &value }, ctl, value.Type()));
  }
};

class buffer_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~buffer_op()
  {}

  buffer_op(
      const std::shared_ptr<const jlm::rvsdg::Type> & type,
      size_t capacity,
      bool pass_through)
      : SimpleOperation({ type }, { type }),
        capacity(capacity),
        pass_through(pass_through)
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const buffer_op *>(&other);
    return ot && ot->capacity == capacity && ot->pass_through == pass_through
        && *ot->result(0) == *result(0);
  }

  std::string
  debug_string() const override
  {
    return util::strfmt("HLS_BUF_", (pass_through ? "P_" : ""), capacity);
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<buffer_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & value, size_t capacity, bool pass_through = false)
  {
    return outputs(
        &rvsdg::CreateOpNode<buffer_op>({ &value }, value.Type(), capacity, pass_through));
  }

  size_t capacity;
  bool pass_through;

private:
};

class triggertype final : public rvsdg::StateType
{
public:
  virtual ~triggertype()
  {}

  triggertype()
      : rvsdg::StateType()
  {}

  std::string
  debug_string() const override
  {
    return "trigger";
  };

  bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override
  {
    auto type = dynamic_cast<const triggertype *>(&other);
    return type;
  };

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  static std::shared_ptr<const triggertype>
  Create();
};

class trigger_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~trigger_op()
  {}

  explicit trigger_op(const std::shared_ptr<const jlm::rvsdg::Type> & type)
      : SimpleOperation({ triggertype::Create(), type }, { type })
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const trigger_op *>(&other);
    // check predicate and value
    return ot && *ot->argument(1) == *argument(1) && *ot->result(0) == *result(0);
  }

  std::string
  debug_string() const override
  {
    return "HLS_TRIGGER";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<trigger_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & tg, jlm::rvsdg::output & value)
  {
    if (!rvsdg::is<triggertype>(tg.Type()))
      throw util::error("Trigger needs to be a triggertype.");

    return outputs(&rvsdg::CreateOpNode<trigger_op>({ &tg, &value }, value.Type()));
  }
};

class print_op final : public rvsdg::SimpleOperation
{
private:
  size_t _id;

public:
  virtual ~print_op()
  {}

  explicit print_op(const std::shared_ptr<const jlm::rvsdg::Type> & type)
      : SimpleOperation({ type }, { type })
  {
    static size_t common_id{ 0 };
    _id = common_id++;
  }

  bool
  operator==(const Operation &) const noexcept override
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

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<print_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & value)
  {
    return outputs(&rvsdg::CreateOpNode<print_op>({ &value }, value.Type()));
  }
};

class loop_op final : public rvsdg::StructuralOperation
{
public:
  virtual ~loop_op() noexcept
  {}

  std::string
  debug_string() const override
  {
    return "HLS_LOOP";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<loop_op>(*this);
  }
};

class backedge_argument;
class backedge_result;
class loop_node;

/**
 * Represents the entry argument for the HLS loop.
 */
class EntryArgument : public rvsdg::RegionArgument
{
  friend loop_node;

public:
  ~EntryArgument() noexcept override;

private:
  EntryArgument(
      rvsdg::Region & region,
      rvsdg::StructuralInput & input,
      const std::shared_ptr<const rvsdg::Type> type)
      : rvsdg::RegionArgument(&region, &input, std::move(type))
  {}

public:
  EntryArgument &
  Copy(rvsdg::Region & region, rvsdg::StructuralInput * input) override;

  // FIXME: This should not be public, but we currently still have some transformations that use
  // this one. Make it eventually private.
  static EntryArgument &
  Create(
      rvsdg::Region & region,
      rvsdg::StructuralInput & input,
      const std::shared_ptr<const rvsdg::Type> type)
  {
    auto argument = new EntryArgument(region, input, std::move(type));
    region.append_argument(argument);
    return *argument;
  }
};

class backedge_argument : public rvsdg::RegionArgument
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

  backedge_argument &
  Copy(rvsdg::Region & region, rvsdg::StructuralInput * input) override;

private:
  backedge_argument(rvsdg::Region * region, const std::shared_ptr<const jlm::rvsdg::Type> & type)
      : rvsdg::RegionArgument(region, nullptr, type),
        result_(nullptr)
  {}

  static backedge_argument *
  create(rvsdg::Region * region, std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    auto argument = new backedge_argument(region, std::move(type));
    region->append_argument(argument);
    return argument;
  }

  backedge_result * result_;
};

class backedge_result : public rvsdg::RegionResult
{
  friend loop_node;
  friend backedge_argument;

public:
  ~backedge_result() override = default;

  backedge_argument *
  argument() const
  {
    return argument_;
  }

  backedge_result &
  Copy(rvsdg::output & origin, rvsdg::StructuralOutput * output) override;

private:
  backedge_result(jlm::rvsdg::output * origin)
      : rvsdg::RegionResult(origin->region(), origin, nullptr, origin->Type()),
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

/**
 * Represents the exit result of the HLS loop.
 */
class ExitResult final : public rvsdg::RegionResult
{
  friend loop_node;

public:
  ~ExitResult() noexcept override;

private:
  ExitResult(rvsdg::output & origin, rvsdg::StructuralOutput & output)
      : rvsdg::RegionResult(origin.region(), &origin, &output, origin.Type())
  {
    JLM_ASSERT(rvsdg::is<loop_op>(origin.region()->node()));
  }

public:
  ExitResult &
  Copy(rvsdg::output & origin, rvsdg::StructuralOutput * output) override;

  // FIXME: This should not be public, but we currently still have some transformations that use
  // this one. Make it eventually private.
  static ExitResult &
  Create(rvsdg::output & origin, rvsdg::StructuralOutput & output)
  {
    auto result = new ExitResult(origin, output);
    origin.region()->append_result(result);
    return *result;
  }
};

class loop_node final : public rvsdg::StructuralNode
{
public:
  virtual ~loop_node()
  {}

private:
  inline loop_node(rvsdg::Region * parent)
      : StructuralNode(parent, 1)
  {}

  jlm::rvsdg::node_output * _predicate_buffer;

public:
  [[nodiscard]] const rvsdg::Operation &
  GetOperation() const noexcept override;

  static loop_node *
  create(rvsdg::Region * parent, bool init = true);

  rvsdg::Region *
  subregion() const noexcept
  {
    return StructuralNode::subregion(0);
  }

  [[nodiscard]] rvsdg::RegionResult *
  predicate() const noexcept
  {
    auto result = subregion()->result(0);
    JLM_ASSERT(dynamic_cast<const rvsdg::ControlType *>(&result->type()));
    return result;
  }

  inline jlm::rvsdg::node_output *
  predicate_buffer() const noexcept
  {
    return _predicate_buffer;
  }

  void
  set_predicate(jlm::rvsdg::output * p);

  backedge_argument *
  add_backedge(std::shared_ptr<const jlm::rvsdg::Type> type);

  rvsdg::StructuralOutput *
  AddLoopVar(jlm::rvsdg::output * origin, jlm::rvsdg::output ** buffer = nullptr);

  jlm::rvsdg::output *
  add_loopconst(jlm::rvsdg::output * origin);

  virtual loop_node *
  copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const override;
};

class bundletype final : public jlm::rvsdg::ValueType
{
public:
  ~bundletype()
  {}

  bundletype(
      const std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> elements)
      : jlm::rvsdg::ValueType(),
        elements_(std::move(elements))
  {}

  bundletype(const bundletype &) = default;

  bundletype(bundletype &&) = delete;

  bundletype &
  operator=(const bundletype &) = delete;

  bundletype &
  operator=(bundletype &&) = delete;

  virtual bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override
  {
    auto type = dynamic_cast<const bundletype *>(&other);
    // TODO: better comparison?
    if (!type || type->elements_.size() != elements_.size())
    {
      return false;
    }
    for (size_t i = 0; i < elements_.size(); ++i)
    {
      if (type->elements_.at(i).first != elements_.at(i).first
          || *type->elements_.at(i).second != *elements_.at(i).second)
      {
        return false;
      }
    }
    return true;
  };

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  std::shared_ptr<const jlm::rvsdg::Type>
  get_element_type(std::string element) const
  {
    for (size_t i = 0; i < elements_.size(); ++i)
    {
      if (elements_.at(i).first == element)
      {
        return elements_.at(i).second;
      }
    }
    // TODO: do something different?
    return {};
  }

  virtual std::string
  debug_string() const override
  {
    return "bundle";
  };

  //        private:
  // TODO: fix memory leak
  const std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> elements_;
};

std::shared_ptr<const bundletype>
get_mem_req_type(std::shared_ptr<const rvsdg::ValueType> elementType, bool write);

std::shared_ptr<const bundletype>
get_mem_res_type(std::shared_ptr<const jlm::rvsdg::ValueType> dataType);

class load_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~load_op()
  {}

  load_op(const std::shared_ptr<const rvsdg::ValueType> & pointeeType, size_t numStates)
      : SimpleOperation(
            CreateInTypes(pointeeType, numStates),
            CreateOutTypes(pointeeType, numStates))
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const load_op *>(&other);
    // check predicate and value
    return ot && *ot->argument(1) == *argument(1) && ot->narguments() == narguments();
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateInTypes(std::shared_ptr<const rvsdg::ValueType> pointeeType, size_t numStates)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types(
        1,
        llvm::PointerType::Create()); // addr
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> states(
        numStates,
        llvm::MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    types.emplace_back(std::move(pointeeType)); // result
    return types;
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateOutTypes(std::shared_ptr<const rvsdg::ValueType> pointeeType, size_t numStates)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types(1, std::move(pointeeType));
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> states(
        numStates,
        llvm::MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    types.emplace_back(llvm::PointerType::Create()); // addr
    return types;
  }

  std::string
  debug_string() const override
  {
    return "HLS_LOAD_" + argument(narguments() - 1)->debug_string();
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<load_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(
      jlm::rvsdg::output & addr,
      const std::vector<jlm::rvsdg::output *> & states,
      jlm::rvsdg::output & load_result)
  {
    std::vector<jlm::rvsdg::output *> inputs;
    inputs.push_back(&addr);
    inputs.insert(inputs.end(), states.begin(), states.end());
    inputs.push_back(&load_result);
    return outputs(&rvsdg::CreateOpNode<load_op>(
        inputs,
        std::dynamic_pointer_cast<const rvsdg::ValueType>(load_result.Type()),
        states.size()));
  }

  [[nodiscard]] const llvm::PointerType &
  GetPointerType() const noexcept
  {
    return *util::AssertedCast<const llvm::PointerType>(argument(0).get());
  }

  [[nodiscard]] std::shared_ptr<const rvsdg::ValueType>
  GetLoadedType() const noexcept
  {
    return std::dynamic_pointer_cast<const rvsdg::ValueType>(result(0));
  }
};

class addr_queue_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~addr_queue_op()
  {}

  addr_queue_op(
      const std::shared_ptr<const llvm::PointerType> & pointerType,
      size_t capacity,
      bool combinatorial)
      : SimpleOperation(CreateInTypes(pointerType), CreateOutTypes(pointerType)),
        combinatorial(combinatorial),
        capacity(capacity)
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const addr_queue_op *>(&other);
    // check predicate and value
    return ot && *ot->argument(1) == *argument(1) && ot->narguments() == narguments();
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateInTypes(std::shared_ptr<const llvm::PointerType> pointerType)
  {
    // check, enq
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types(2, std::move(pointerType));
    types.emplace_back(llvm::MemoryStateType::Create()); // deq
    return types;
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateOutTypes(std::shared_ptr<const llvm::PointerType> pointerType)
  {
    return { std::move(pointerType) };
  }

  std::string
  debug_string() const override
  {
    if (combinatorial)
    {
      return "HLS_ADDR_QUEUE_COMB_" + argument(narguments() - 1)->debug_string();
    }
    return "HLS_ADDR_QUEUE_" + argument(narguments() - 1)->debug_string();
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<addr_queue_op>(*this);
  }

  static jlm::rvsdg::output *
  create(
      jlm::rvsdg::output & check,
      jlm::rvsdg::output & enq,
      jlm::rvsdg::output & deq,
      bool combinatorial,
      size_t capacity = 10)
  {
    return rvsdg::CreateOpNode<addr_queue_op>(
               { &check, &enq, &deq },
               std::dynamic_pointer_cast<const llvm::PointerType>(check.Type()),
               capacity,
               combinatorial)
        .output(0);
  }

  bool combinatorial;
  size_t capacity;
};

class state_gate_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~state_gate_op()
  {}

  state_gate_op(const std::shared_ptr<const jlm::rvsdg::Type> & type, size_t numStates)
      : SimpleOperation(CreateInOutTypes(type, numStates), CreateInOutTypes(type, numStates))
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const state_gate_op *>(&other);
    // check predicate and value
    return ot && *ot->argument(1) == *argument(1) && ot->narguments() == narguments();
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateInOutTypes(const std::shared_ptr<const jlm::rvsdg::Type> & type, size_t numStates)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types(1, type);
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> states(
        numStates,
        llvm::MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    return types;
  }

  std::string
  debug_string() const override
  {
    return "HLS_STATE_GATE_" + argument(narguments() - 1)->debug_string();
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<state_gate_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & addr, const std::vector<jlm::rvsdg::output *> & states)
  {
    std::vector<jlm::rvsdg::output *> inputs;
    inputs.push_back(&addr);
    inputs.insert(inputs.end(), states.begin(), states.end());
    return outputs(&rvsdg::CreateOpNode<state_gate_op>(inputs, addr.Type(), states.size()));
  }
};

class decoupled_load_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~decoupled_load_op()
  {}

  decoupled_load_op(const std::shared_ptr<const rvsdg::ValueType> & pointeeType, size_t capacity)
      : SimpleOperation(CreateInTypes(pointeeType), CreateOutTypes(pointeeType)),
        capacity(capacity)
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const decoupled_load_op *>(&other);
    // check predicate and value
    return ot && *ot->argument(1) == *argument(1) && ot->narguments() == narguments();
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateInTypes(std::shared_ptr<const rvsdg::ValueType> pointeeType)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types(1, llvm::PointerType::Create());
    types.emplace_back(std::move(pointeeType)); // result
    return types;
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateOutTypes(std::shared_ptr<const rvsdg::ValueType> pointeeType)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types(1, std::move(pointeeType));
    types.emplace_back(llvm::PointerType::Create()); // addr
    return types;
  }

  std::string
  debug_string() const override
  {
    return "HLS_DEC_LOAD_" + std::to_string(capacity) + "_"
         + argument(narguments() - 1)->debug_string();
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<decoupled_load_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & addr, jlm::rvsdg::output & load_result, size_t capacity)
  {
    std::vector<jlm::rvsdg::output *> inputs;
    inputs.push_back(&addr);
    inputs.push_back(&load_result);
    JLM_ASSERT(capacity >= 1);
    return outputs(&rvsdg::CreateOpNode<decoupled_load_op>(
        inputs,
        std::dynamic_pointer_cast<const rvsdg::ValueType>(load_result.Type()),
        capacity));
  }

  [[nodiscard]] const llvm::PointerType &
  GetPointerType() const noexcept
  {
    return *util::AssertedCast<const llvm::PointerType>(argument(0).get());
  }

  [[nodiscard]] std::shared_ptr<const rvsdg::ValueType>
  GetLoadedType() const noexcept
  {
    return std::dynamic_pointer_cast<const rvsdg::ValueType>(result(0));
  }

  size_t capacity;
};

class mem_resp_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~mem_resp_op()
  {}

  explicit mem_resp_op(
      const std::vector<std::shared_ptr<const rvsdg::Type>> & output_types,
      int in_width)
      : SimpleOperation(CreateInTypes(in_width), CreateOutTypes(output_types))
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const mem_resp_op *>(&other);
    // check predicate and value
    return ot && *ot->argument(1) == *argument(1) && ot->narguments() == narguments();
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateInTypes(int in_width)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types;
    types.emplace_back(get_mem_res_type(jlm::rvsdg::bittype::Create(in_width)));
    return types;
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateOutTypes(const std::vector<std::shared_ptr<const rvsdg::Type>> & output_types)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types;
    types.reserve(output_types.size());
    for (auto outputType : output_types)
    {
      types.emplace_back(outputType);
    }
    return types;
  }

  std::string
  debug_string() const override
  {
    return "HLS_MEM_RESP";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<mem_resp_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(
      rvsdg::output & result,
      const std::vector<std::shared_ptr<const rvsdg::Type>> & output_types,
      int in_width)
  {
    return outputs(&rvsdg::CreateOpNode<mem_resp_op>({ &result }, output_types, in_width));
  }
};

class mem_req_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~mem_req_op() = default;

  mem_req_op(
      const std::vector<std::shared_ptr<const rvsdg::ValueType>> & load_types,
      const std::vector<std::shared_ptr<const rvsdg::ValueType>> & store_types)
      : SimpleOperation(
            CreateInTypes(load_types, store_types),
            CreateOutTypes(load_types, store_types))
  {
    for (auto loadType : load_types)
    {
      LoadTypes_.emplace_back(loadType);
    }
    for (auto storeType : store_types)
    {
      StoreTypes_.emplace_back(storeType);
    }
  }

  mem_req_op(const mem_req_op & other) = default;

  bool
  operator==(const Operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const mem_req_op *>(&other);
    // check predicate and value
    return ot && ot->narguments() == narguments()
        && (ot->narguments() == 0 || (*ot->argument(1) == *argument(1)))
        && ot->narguments() == narguments();
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateInTypes(
      const std::vector<std::shared_ptr<const rvsdg::ValueType>> & load_types,
      const std::vector<std::shared_ptr<const rvsdg::ValueType>> & store_types)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types;
    for (size_t i = 0; i < load_types.size(); i++)
    {
      types.emplace_back(llvm::PointerType::Create()); // addr
    }
    for (auto storeType : store_types)
    {
      types.emplace_back(llvm::PointerType::Create()); // addr
      types.emplace_back(storeType);                   // data
    }
    return types;
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateOutTypes(
      const std::vector<std::shared_ptr<const rvsdg::ValueType>> & load_types,
      const std::vector<std::shared_ptr<const rvsdg::ValueType>> & store_types)
  {
    int max_width = 0;
    for (auto tp : load_types)
    {
      auto sz = JlmSize(tp.get());
      max_width = sz > max_width ? sz : max_width;
    }
    for (auto tp : store_types)
    {
      auto sz = JlmSize(tp.get());
      max_width = sz > max_width ? sz : max_width;
    }
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types;
    types.emplace_back(
        get_mem_req_type(jlm::rvsdg::bittype::Create(max_width), !store_types.empty()));
    return types;
  }

  std::string
  debug_string() const override
  {
    return "HLS_MEM_REQ";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<mem_req_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(
      const std::vector<jlm::rvsdg::output *> & load_operands,
      const std::vector<std::shared_ptr<const rvsdg::ValueType>> & loadTypes,
      const std::vector<jlm::rvsdg::output *> & store_operands,
      rvsdg::Region *)
  {
    // Stores have both addr and data operand
    // But we are only interested in the data operand type
    JLM_ASSERT(store_operands.size() % 2 == 0);
    std::vector<std::shared_ptr<const rvsdg::ValueType>> storeTypes;
    for (size_t i = 1; i < store_operands.size(); i += 2)
    {
      storeTypes.push_back(
          std::dynamic_pointer_cast<const rvsdg::ValueType>(store_operands[i]->Type()));
    }
    std::vector operands(load_operands);
    operands.insert(operands.end(), store_operands.begin(), store_operands.end());
    return outputs(&rvsdg::CreateOpNode<mem_req_op>(operands, loadTypes, storeTypes));
  }

  size_t
  get_nloads() const
  {
    return LoadTypes_.size();
  }

  const std::vector<std::shared_ptr<const rvsdg::Type>> *
  GetLoadTypes() const
  {
    return &LoadTypes_;
  }

  const std::vector<std::shared_ptr<const rvsdg::Type>> *
  GetStoreTypes() const
  {
    return &StoreTypes_;
  }

private:
  std::vector<std::shared_ptr<const rvsdg::Type>> LoadTypes_;
  std::vector<std::shared_ptr<const rvsdg::Type>> StoreTypes_;
};

class store_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~store_op()
  {}

  store_op(const std::shared_ptr<const rvsdg::ValueType> & pointeeType, size_t numStates)
      : SimpleOperation(
            CreateInTypes(pointeeType, numStates),
            CreateOutTypes(pointeeType, numStates))
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const store_op *>(&other);
    // check predicate and value
    return ot && *ot->argument(1) == *argument(1) && ot->narguments() == narguments();
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateInTypes(const std::shared_ptr<const rvsdg::ValueType> & pointeeType, size_t numStates)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types(
        { llvm::PointerType::Create(), pointeeType });
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> states(
        numStates + 1,
        llvm::MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    return types;
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateOutTypes(const std::shared_ptr<const rvsdg::ValueType> & pointeeType, size_t numStates)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types(
        numStates,
        llvm::MemoryStateType::Create());
    types.emplace_back(llvm::PointerType::Create()); // addr
    types.emplace_back(pointeeType);                 // data
    return types;
  }

  std::string
  debug_string() const override
  {
    return "HLS_STORE_" + argument(narguments() - 1)->debug_string();
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<store_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(
      jlm::rvsdg::output & addr,
      jlm::rvsdg::output & value,
      const std::vector<jlm::rvsdg::output *> & states,
      jlm::rvsdg::output & resp)
  {
    std::vector<jlm::rvsdg::output *> inputs;
    inputs.push_back(&addr);
    inputs.push_back(&value);
    inputs.insert(inputs.end(), states.begin(), states.end());
    inputs.push_back(&resp);
    return outputs(&rvsdg::CreateOpNode<store_op>(
        inputs,
        std::dynamic_pointer_cast<const rvsdg::ValueType>(value.Type()),
        states.size()));
  }

  [[nodiscard]] const llvm::PointerType &
  GetPointerType() const noexcept
  {
    return *util::AssertedCast<const llvm::PointerType>(argument(0).get());
  }

  [[nodiscard]] const rvsdg::ValueType &
  GetStoredType() const noexcept
  {
    return *util::AssertedCast<const rvsdg::ValueType>(argument(1).get());
  }
};

class local_mem_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~local_mem_op()
  {}

  explicit local_mem_op(std::shared_ptr<const llvm::ArrayType> at)
      : SimpleOperation({}, CreateOutTypes(std::move(at)))
  {}

  bool
  operator==(const Operation &) const noexcept override
  {
    // TODO:
    // auto ot = dynamic_cast<const local_mem_op *>(&other);
    // check predicate and value
    return false;
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateOutTypes(std::shared_ptr<const llvm::ArrayType> at)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types(2, std::move(at));
    return types;
  }

  std::string
  debug_string() const override
  {
    return "HLS_LOCAL_MEM_" + result(0)->debug_string();
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<local_mem_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(std::shared_ptr<const llvm::ArrayType> at, rvsdg::Region * region)
  {
    return outputs(&rvsdg::CreateOpNode<local_mem_op>(*region, std::move(at)));
  }
};

class local_mem_resp_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~local_mem_resp_op()
  {}

  local_mem_resp_op(const std::shared_ptr<const llvm::ArrayType> & at, size_t resp_count)
      : SimpleOperation({ at }, CreateOutTypes(at, resp_count))
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const local_mem_resp_op *>(&other);
    // check predicate and value
    return ot && *ot->argument(1) == *argument(1) && ot->narguments() == narguments();
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateOutTypes(const std::shared_ptr<const jlm::llvm::ArrayType> & at, size_t resp_count)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types(resp_count, at->GetElementType());
    return types;
  }

  std::string
  debug_string() const override
  {
    return "HLS_LOCAL_MEM_RESP";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<local_mem_resp_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & mem, size_t resp_count)
  {
    return outputs(&rvsdg::CreateOpNode<local_mem_resp_op>(
        { &mem },
        std::dynamic_pointer_cast<const llvm::ArrayType>(mem.Type()),
        resp_count));
  }
};

class local_load_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~local_load_op()
  {}

  local_load_op(const std::shared_ptr<const jlm::rvsdg::ValueType> & valuetype, size_t numStates)
      : SimpleOperation(CreateInTypes(valuetype, numStates), CreateOutTypes(valuetype, numStates))
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const local_load_op *>(&other);
    // check predicate and value
    return ot && *ot->argument(1) == *argument(1) && ot->narguments() == narguments();
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateInTypes(const std::shared_ptr<const jlm::rvsdg::ValueType> & valuetype, size_t numStates)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types(1, jlm::rvsdg::bittype::Create(64));
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> states(
        numStates,
        llvm::MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    types.emplace_back(valuetype); // result
    return types;
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateOutTypes(const std::shared_ptr<const jlm::rvsdg::ValueType> & valuetype, size_t numStates)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types(1, valuetype);
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> states(
        numStates,
        llvm::MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    types.emplace_back(jlm::rvsdg::bittype::Create(64)); // addr
    return types;
  }

  std::string
  debug_string() const override
  {
    return "HLS_LOCAL_LOAD_" + argument(narguments() - 1)->debug_string();
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<local_load_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(
      jlm::rvsdg::output & index,
      const std::vector<jlm::rvsdg::output *> & states,
      jlm::rvsdg::output & load_result)
  {
    std::vector<jlm::rvsdg::output *> inputs;
    inputs.push_back(&index);
    inputs.insert(inputs.end(), states.begin(), states.end());
    inputs.push_back(&load_result);
    return outputs(&rvsdg::CreateOpNode<local_load_op>(
        inputs,
        std::dynamic_pointer_cast<const jlm::rvsdg::ValueType>(load_result.Type()),
        states.size()));
  }

  [[nodiscard]] std::shared_ptr<const rvsdg::ValueType>
  GetLoadedType() const noexcept
  {
    return std::dynamic_pointer_cast<const rvsdg::ValueType>(result(0));
  }
};

class local_store_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~local_store_op()
  {}

  local_store_op(const std::shared_ptr<const jlm::rvsdg::ValueType> & valuetype, size_t numStates)
      : SimpleOperation(CreateInTypes(valuetype, numStates), CreateOutTypes(valuetype, numStates))
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const local_store_op *>(&other);
    // check predicate and value
    return ot && *ot->argument(1) == *argument(1) && ot->narguments() == narguments();
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateInTypes(const std::shared_ptr<const jlm::rvsdg::ValueType> & valuetype, size_t numStates)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types(
        { jlm::rvsdg::bittype::Create(64), valuetype });
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> states(
        numStates,
        llvm::MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    return types;
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateOutTypes(const std::shared_ptr<const jlm::rvsdg::ValueType> & valuetype, size_t numStates)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types(
        numStates,
        llvm::MemoryStateType::Create());
    types.emplace_back(jlm::rvsdg::bittype::Create(64)); // addr
    types.emplace_back(valuetype);                       // data
    return types;
  }

  std::string
  debug_string() const override
  {
    return "HLS_LOCAL_STORE_" + argument(narguments() - 1)->debug_string();
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<local_store_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(
      jlm::rvsdg::output & index,
      jlm::rvsdg::output & value,
      const std::vector<jlm::rvsdg::output *> & states)
  {
    std::vector<jlm::rvsdg::output *> inputs;
    inputs.push_back(&index);
    inputs.push_back(&value);
    inputs.insert(inputs.end(), states.begin(), states.end());
    return outputs(&rvsdg::CreateOpNode<local_store_op>(
        inputs,
        std::dynamic_pointer_cast<const jlm::rvsdg::ValueType>(value.Type()),
        states.size()));
  }

  [[nodiscard]] const jlm::rvsdg::ValueType &
  GetStoredType() const noexcept
  {
    return *util::AssertedCast<const jlm::rvsdg::ValueType>(argument(1).get());
  }
};

class local_mem_req_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~local_mem_req_op()
  {}

  local_mem_req_op(
      const std::shared_ptr<const llvm::ArrayType> & at,
      size_t load_cnt,
      size_t store_cnt)
      : SimpleOperation(CreateInTypes(at, load_cnt, store_cnt), {})
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const local_mem_req_op *>(&other);
    // check predicate and value
    return ot && ot->narguments() == narguments()
        && (ot->narguments() == 0 || (*ot->argument(1) == *argument(1)))
        && ot->narguments() == narguments();
  }

  static std::vector<std::shared_ptr<const jlm::rvsdg::Type>>
  CreateInTypes(
      const std::shared_ptr<const llvm::ArrayType> & at,
      size_t load_cnt,
      size_t store_cnt)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types(1, at);
    for (size_t i = 0; i < load_cnt; ++i)
    {
      types.emplace_back(jlm::rvsdg::bittype::Create(64)); // addr
    }
    for (size_t i = 0; i < store_cnt; ++i)
    {
      types.emplace_back(jlm::rvsdg::bittype::Create(64)); // addr
      types.emplace_back(at->GetElementType());            // data
    }
    return types;
  }

  std::string
  debug_string() const override
  {
    return "HLS_LOCAL_MEM_REQ";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<local_mem_req_op>(*this);
  }

  static std::vector<jlm::rvsdg::output *>
  create(
      jlm::rvsdg::output & mem,
      const std::vector<jlm::rvsdg::output *> & load_operands,
      const std::vector<jlm::rvsdg::output *> & store_operands)
  {
    JLM_ASSERT(store_operands.size() % 2 == 0);
    std::vector operands(1, &mem);
    operands.insert(operands.end(), load_operands.begin(), load_operands.end());
    operands.insert(operands.end(), store_operands.begin(), store_operands.end());
    return outputs(&rvsdg::CreateOpNode<local_mem_req_op>(
        operands,
        std::dynamic_pointer_cast<const llvm::ArrayType>(mem.Type()),
        load_operands.size(),
        store_operands.size() / 2));
  }
};

}
#endif // JLM_HLS_IR_HLS_HPP

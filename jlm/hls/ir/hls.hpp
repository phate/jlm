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
#include <jlm/rvsdg/type.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/util/common.hpp>
#include <memory>
#include <utility>

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

class loop_constant_buffer_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~loop_constant_buffer_op()
  {}

  loop_constant_buffer_op(const jlm::rvsdg::ctltype & ctltype, const jlm::rvsdg::type & type)
      : jlm::rvsdg::simple_op({ ctltype, type }, { type })
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const loop_constant_buffer_op *>(&other);
    return ot && ot->result(0).type() == result(0).type()
        && ot->argument(0).type() == argument(0).type();
  }

  std::string
  debug_string() const override
  {
    return "HLS_LOOP_CONST_BUF";
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new loop_constant_buffer_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & predicate, jlm::rvsdg::output & value)
  {
    auto region = predicate.region();
    auto ctl = dynamic_cast<const jlm::rvsdg::ctltype *>(&predicate.type());
    if (!ctl)
      throw util::error("Predicate needs to be a ctltype.");
    loop_constant_buffer_op op(*ctl, value.type());
    return jlm::rvsdg::simple_node::create_normalized(region, op, { &predicate, &value });
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
  argument() const
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

  jlm::rvsdg::output *
  add_loopconst(jlm::rvsdg::output * origin);

  virtual loop_node *
  copy(jlm::rvsdg::region * region, jlm::rvsdg::substitution_map & smap) const override;
};

class bundletype final : public jlm::rvsdg::valuetype
{
public:
  ~bundletype()
  {}

  bundletype(const std::vector<std::pair<std::string, std::unique_ptr<jlm::rvsdg::type>>> * elements)
      : jlm::rvsdg::valuetype(),
        elements_(std::move(elements))
  {}

  bundletype(const bundletype &) = default;

  bundletype(bundletype &&) = delete;

  bundletype &
  operator=(const bundletype &) = delete;

  bundletype &
  operator=(bundletype &&) = delete;

  virtual bool
  operator==(const jlm::rvsdg::type & other) const noexcept override
  {
    auto type = dynamic_cast<const bundletype *>(&other);
    // TODO: better comparison?
    if (!type || type->elements_->size() != elements_->size())
    {
      return false;
    }
    for (size_t i = 0; i < elements_->size(); ++i)
    {
      if (type->elements_->at(i).first != elements_->at(i).first
          || *type->elements_->at(i).second != *elements_->at(i).second)
      {
        return false;
      }
    }
    return true;
  };

  jlm::rvsdg::type *
  get_element_type(std::string element) const
  {
    for (size_t i = 0; i < elements_->size(); ++i)
    {
      if (elements_->at(i).first == element)
      {
        return elements_->at(i).second.get();
      }
    }
    // TODO: do something different?
    return nullptr;
  }

  virtual std::unique_ptr<jlm::rvsdg::type>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::type>(new bundletype(*this));
  }

  virtual std::string
  debug_string() const override
  {
    return "bundle";
  };

  //        private:
  // TODO: fix memory leak
  const std::vector<std::pair<std::string, std::unique_ptr<jlm::rvsdg::type>>> * elements_;
};

std::unique_ptr<bundletype>
get_mem_req_type(const rvsdg::valuetype & elementType, bool write);

std::unique_ptr<bundletype>
get_mem_res_type(const jlm::rvsdg::valuetype & dataType);

class load_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~load_op()
  {}

  load_op(const rvsdg::valuetype & pointeeType, size_t numStates)
      : simple_op(CreateInPorts(pointeeType, numStates), CreateOutPorts(pointeeType, numStates))
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const load_op *>(&other);
    // check predicate and value
    return ot && ot->argument(1).type() == argument(1).type() && ot->narguments() == narguments();
  }

  static std::vector<jlm::rvsdg::port>
  CreateInPorts(const rvsdg::valuetype & pointeeType, size_t numStates)
  {
    std::vector<jlm::rvsdg::port> ports(1, { llvm::PointerType() });
    //                ports.emplace_back(pointerType.GetElementType());
    std::vector<jlm::rvsdg::port> states(numStates, { llvm::MemoryStateType::Create() });
    ports.insert(ports.end(), states.begin(), states.end());
    ports.emplace_back(pointeeType); // result
    return ports;
  }

  static std::vector<jlm::rvsdg::port>
  CreateOutPorts(const rvsdg::valuetype & pointeeType, size_t numStates)
  {
    std::vector<jlm::rvsdg::port> ports(1, { pointeeType });
    std::vector<jlm::rvsdg::port> states(numStates, { llvm::MemoryStateType::Create() });
    ports.insert(ports.end(), states.begin(), states.end());
    ports.emplace_back(llvm::PointerType()); // addr
    return ports;
  }

  std::string
  debug_string() const override
  {
    return "HLS_LOAD_" + argument(narguments() - 1).type().debug_string();
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new load_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(
      jlm::rvsdg::output & addr,
      const std::vector<jlm::rvsdg::output *> & states,
      jlm::rvsdg::output & load_result)
  {
    auto region = addr.region();
    auto pointerType = dynamic_cast<const llvm::PointerType *>(&addr.type());
    load_op op(*pointerType, states.size());
    std::vector<jlm::rvsdg::output *> inputs;
    inputs.push_back(&addr);
    inputs.insert(inputs.end(), states.begin(), states.end());
    inputs.push_back(&load_result);
    return jlm::rvsdg::simple_node::create_normalized(region, op, inputs);
  }
};

class addr_queue_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~addr_queue_op()
  {}

  addr_queue_op(const llvm::PointerType & pointerType, size_t capacity)
      : simple_op(CreateInPorts(pointerType), CreateOutPorts(pointerType)),
        capacity(capacity)
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const addr_queue_op *>(&other);
    // check predicate and value
    return ot && ot->argument(1).type() == argument(1).type() && ot->narguments() == narguments();
  }

  static std::vector<jlm::rvsdg::port>
  CreateInPorts(const llvm::PointerType & pointerType)
  {
    // check, enq
    std::vector<jlm::rvsdg::port> ports(2, { pointerType });
    ports.emplace_back(llvm::MemoryStateType::Create()); // deq
    return ports;
  }

  static std::vector<jlm::rvsdg::port>
  CreateOutPorts(const llvm::PointerType & pointerType)
  {
    return { pointerType };
  }

  std::string
  debug_string() const override
  {
    return "HLS_ADDR_QUEUE_" + argument(narguments() - 1).type().debug_string();
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new addr_queue_op(*this));
  }

  static jlm::rvsdg::output *
  create(jlm::rvsdg::output & check, jlm::rvsdg::output & enq, jlm::rvsdg::output & deq, size_t capacity = 10)
  {
    auto region = check.region();
    auto pointerType = dynamic_cast<const llvm::PointerType *>(&check.type());
    addr_queue_op op(*pointerType, capacity);
    return jlm::rvsdg::simple_node::create_normalized(region, op, { &check, &enq, &deq })[0];
  }

  size_t capacity;
};

class state_gate_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~state_gate_op()
  {}

  state_gate_op(const jlm::rvsdg::type & type, size_t numStates)
      : simple_op(CreateInOutPorts(type, numStates), CreateInOutPorts(type, numStates))
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const state_gate_op *>(&other);
    // check predicate and value
    return ot && ot->argument(1).type() == argument(1).type() && ot->narguments() == narguments();
  }

  static std::vector<jlm::rvsdg::port>
  CreateInOutPorts(const jlm::rvsdg::type & type, size_t numStates)
  {
    std::vector<jlm::rvsdg::port> ports(1, { type });
    std::vector<jlm::rvsdg::port> states(numStates, { llvm::MemoryStateType::Create() });
    ports.insert(ports.end(), states.begin(), states.end());
    return ports;
  }

  std::string
  debug_string() const override
  {
    return "HLS_STATE_GATE_" + argument(narguments() - 1).type().debug_string();
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new state_gate_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & addr, const std::vector<jlm::rvsdg::output *> & states)
  {
    auto region = addr.region();
    //                auto pointerType = dynamic_cast<const llvm::PointerType*>(&addr.type());
    state_gate_op op(addr.type(), states.size());
    std::vector<jlm::rvsdg::output *> inputs;
    inputs.push_back(&addr);
    inputs.insert(inputs.end(), states.begin(), states.end());
    return jlm::rvsdg::simple_node::create_normalized(region, op, inputs);
  }
};

class decoupled_load_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~decoupled_load_op()
  {}

  decoupled_load_op(const rvsdg::valuetype & pointeeType)
      : simple_op(CreateInPorts(pointeeType), CreateOutPorts(pointeeType))
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const decoupled_load_op *>(&other);
    // check predicate and value
    return ot && ot->argument(1).type() == argument(1).type() && ot->narguments() == narguments();
  }

  static std::vector<jlm::rvsdg::port>
  CreateInPorts(const rvsdg::valuetype & pointeeType)
  {
    std::vector<jlm::rvsdg::port> ports(1, { llvm::PointerType() });
    ports.emplace_back(pointeeType); // result
    return ports;
  }

  static std::vector<jlm::rvsdg::port>
  CreateOutPorts(const rvsdg::valuetype & pointeeType)
  {
    std::vector<jlm::rvsdg::port> ports(1, {pointeeType });
    ports.emplace_back(llvm::PointerType()); // addr
    return ports;
  }

  std::string
  debug_string() const override
  {
    return "HLS_DEC_LOAD_" + argument(narguments() - 1).type().debug_string();
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new decoupled_load_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & addr, jlm::rvsdg::output & load_result)
  {
    auto region = addr.region();
    auto pointerType = dynamic_cast<const llvm::PointerType *>(&addr.type());
    decoupled_load_op op(*pointerType);
    std::vector<jlm::rvsdg::output *> inputs;
    inputs.push_back(&addr);
    inputs.push_back(&load_result);
    return jlm::rvsdg::simple_node::create_normalized(region, op, inputs);
  }
};

class mem_resp_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~mem_resp_op()
  {}

  mem_resp_op(const std::vector<const jlm::rvsdg::type *> & output_types)
      : simple_op(CreateInPorts(output_types), CreateOutPorts(output_types))
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const mem_resp_op *>(&other);
    // check predicate and value
    return ot && ot->argument(1).type() == argument(1).type() && ot->narguments() == narguments();
  }

  static std::vector<jlm::rvsdg::port>
  CreateInPorts(const std::vector<const jlm::rvsdg::type *> & output_types)
  {
    size_t max_width = 64;
    // TODO: calculate size onece JlmSize is moved
    //                size_t max_width = 0;
    //                for (auto tp:output_types) {
    //                    auto sz = jlm::hls::BaseHLS::JlmSize(tp);
    //                    max_width = sz>max_width?sz:max_width;
    //                }
    std::vector<jlm::rvsdg::port> ports;
    ports.emplace_back(get_mem_res_type(jlm::rvsdg::bittype(max_width)));
    return ports;
  }

  static std::vector<jlm::rvsdg::port>
  CreateOutPorts(const std::vector<const jlm::rvsdg::type *> & output_types)
  {
    std::vector<jlm::rvsdg::port> ports;
    ports.reserve(output_types.size());
    for (auto t : output_types)
    {
      ports.emplace_back(*t);
    }
    return ports;
  }

  std::string
  debug_string() const override
  {
    return "HLS_MEM_RESP";
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new mem_resp_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & result, const std::vector<const jlm::rvsdg::type *> & output_types)
  {
    auto region = result.region();
    // TODO: verify port here
    //                auto result_type = dynamic_cast<const jlm::rvsdg::bittype*>(&result.type());
    //                JLM_ASSERT(result_type && result_type->nbits()==64);
    mem_resp_op op(output_types);
    return jlm::rvsdg::simple_node::create_normalized(region, op, { &result });
  }
};

class mem_req_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~mem_req_op()
  {}

  mem_req_op(
      const std::vector<const llvm::PointerType *> & load_types,
      const std::vector<const rvsdg::valuetype *> & store_types)
      : simple_op(CreateInPorts(load_types, store_types), CreateOutPorts(load_types, store_types)),
        nloads(load_types.size())
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const mem_req_op *>(&other);
    // check predicate and value
    return ot && ot->narguments() == narguments()
        && (ot->narguments() == 0 || (ot->argument(1).type() == argument(1).type()))
        && ot->narguments() == narguments();
  }

  static std::vector<jlm::rvsdg::port>
  CreateInPorts(
      const std::vector<const llvm::PointerType *> & load_types,
      const std::vector<const rvsdg::valuetype *> & store_types)
  {
    std::vector<jlm::rvsdg::port> ports;
    for (auto lt : load_types)
    {
      ports.emplace_back(*lt); // addr
    }
    for (auto st : store_types)
    {
      ports.emplace_back(llvm::PointerType());                  // addr
      ports.emplace_back(*st); // data
    }
    return ports;
}

  static std::vector<jlm::rvsdg::port>
  CreateOutPorts(
      const std::vector<const llvm::PointerType *> & load_types,
      const std::vector<const rvsdg::valuetype *> & store_types)
  {
    size_t max_width = 64;
    // TODO: fix once JlmSize is moved
    //                size_t max_width = 0;
    //                for (auto tp:load_types) {
    //                    auto sz = jlm::hls::BaseHLS::JlmSize(tp);
    //                    max_width = sz>max_width?sz:max_width;
    //                }
    //                for (auto tp:store_types) {
    //                    auto sz = jlm::hls::BaseHLS::JlmSize(tp);
    //                    max_width = sz>max_width?sz:max_width;
    //                }
    std::vector<jlm::rvsdg::port> ports;
    ports.emplace_back(
        get_mem_req_type(jlm::rvsdg::bittype(max_width), !store_types.empty()));
    return ports;
  }

  std::string
  debug_string() const override
  {
    return "HLS_MEM_REQ";
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new mem_req_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(
      const std::vector<jlm::rvsdg::output *> & load_operands,
      const std::vector<jlm::rvsdg::output *> & store_operands,
      jlm::rvsdg::region * region)
  {
    std::vector<const llvm::PointerType *> load_types;
    for (size_t i = 0; i < load_operands.size(); ++i)
    {
      auto pt = dynamic_cast<const llvm::PointerType *>(&load_operands[i]->type());
      JLM_ASSERT(pt);
      load_types.push_back(pt);
    }
    JLM_ASSERT(store_operands.size() % 2 == 0);
    std::vector<const rvsdg::valuetype *> store_types;
    for (size_t i = 1; i < store_operands.size(); i += 2)
    {
      auto pt = dynamic_cast<const rvsdg::valuetype *>(&store_operands[i]->type());
//      auto et = &store_operands[i + 1]->type();
//      JLM_ASSERT(pt && et && pt->GetElementType() == *et);
      store_types.push_back(pt);
    }
    mem_req_op op(load_types, store_types);
    std::vector<jlm::rvsdg::output *> operands(load_operands);
    operands.insert(operands.end(), store_operands.begin(), store_operands.end());
    return jlm::rvsdg::simple_node::create_normalized(region, op, operands);
  }

  size_t
  get_nloads() const
  {
    return nloads;
  }

private:
  size_t nloads;
};

class store_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~store_op()
  {}

  store_op(const rvsdg::valuetype & pointeeType, size_t numStates)
      : simple_op(CreateInPorts(pointeeType, numStates), CreateOutPorts(pointeeType, numStates))
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const store_op *>(&other);
    // check predicate and value
    return ot && ot->argument(1).type() == argument(1).type() && ot->narguments() == narguments();
  }

  static std::vector<jlm::rvsdg::port>
  CreateInPorts(const rvsdg::valuetype & pointeeType, size_t numStates)
  {
    std::vector<jlm::rvsdg::port> ports({ llvm::PointerType(), pointeeType });
    std::vector<jlm::rvsdg::port> states(numStates, { llvm::MemoryStateType::Create() });
    ports.insert(ports.end(), states.begin(), states.end());
    return ports;
  }

  static std::vector<jlm::rvsdg::port>
  CreateOutPorts(const rvsdg::valuetype & pointeeType, size_t numStates)
  {
    std::vector<jlm::rvsdg::port> ports(numStates, { llvm::MemoryStateType::Create() });
    ports.emplace_back(llvm::PointerType());                  // addr
    ports.emplace_back(pointeeType); // data
    return ports;
  }

  std::string
  debug_string() const override
  {
    return "HLS_STORE_" + argument(narguments() - 1).type().debug_string();
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new store_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & addr, jlm::rvsdg::output & value, const std::vector<jlm::rvsdg::output *> & states)
  {
    auto region = addr.region();
    auto pointerType = dynamic_cast<const llvm::PointerType *>(&addr.type());
    store_op op(*pointerType, states.size());
    std::vector<jlm::rvsdg::output *> inputs;
    inputs.push_back(&addr);
    inputs.push_back(&value);
    inputs.insert(inputs.end(), states.begin(), states.end());
    return jlm::rvsdg::simple_node::create_normalized(region, op, inputs);
  }
};

class local_mem_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~local_mem_op()
  {}

  local_mem_op(const llvm::arraytype & at)
      : simple_op({}, CreateOutPorts(at))
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const local_mem_op *>(&other);
    // check predicate and value
    return false;
  }

  static std::vector<jlm::rvsdg::port>
  CreateOutPorts(const llvm::arraytype & at)
  {
    std::vector<jlm::rvsdg::port> ports(2, { at });
    return ports;
  }

  std::string
  debug_string() const override
  {
    return "HLS_LOCAL_MEM_" + result(0).type().debug_string();
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new local_mem_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(const jlm::llvm::arraytype &at, jlm::rvsdg::region * region)
  {
    local_mem_op op(at);
    return jlm::rvsdg::simple_node::create_normalized(region, op, {});
  }
};

class local_mem_resp_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~local_mem_resp_op()
  {}

  local_mem_resp_op(const jlm::llvm::arraytype & at, size_t resp_count)
      : simple_op({ at }, CreateOutPorts(at, resp_count))
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const local_mem_resp_op *>(&other);
    // check predicate and value
    return ot && ot->argument(1).type() == argument(1).type() && ot->narguments() == narguments();
  }

  static std::vector<jlm::rvsdg::port>
  CreateOutPorts(const jlm::llvm::arraytype & at, size_t resp_count)
  {
    std::vector<jlm::rvsdg::port> ports(resp_count, at.element_type());
    return ports;
  }

  std::string
  debug_string() const override
  {
    return "HLS_LOCAL_MEM_RESP";
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new local_mem_resp_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & mem, size_t resp_count)
  {
    auto region = mem.region();
    auto at = dynamic_cast<const jlm::llvm::arraytype *>(&mem.type());
    local_mem_resp_op op(*at, resp_count);
    return jlm::rvsdg::simple_node::create_normalized(region, op, { &mem });
  }
};

class local_load_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~local_load_op()
  {}

  local_load_op(const jlm::rvsdg::valuetype & valuetype, size_t numStates)
      : simple_op(CreateInPorts(valuetype, numStates), CreateOutPorts(valuetype, numStates))
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const local_load_op *>(&other);
    // check predicate and value
    return ot && ot->argument(1).type() == argument(1).type() && ot->narguments() == narguments();
  }

  static std::vector<jlm::rvsdg::port>
  CreateInPorts(const jlm::rvsdg::valuetype & valuetype, size_t numStates)
  {
    std::vector<jlm::rvsdg::port> ports(1, { jlm::rvsdg::bit64 });
    //                ports.emplace_back(pointerType.GetElementType());
    std::vector<jlm::rvsdg::port> states(numStates, { llvm::MemoryStateType::Create() });
    ports.insert(ports.end(), states.begin(), states.end());
    ports.emplace_back(valuetype); // result
    return ports;
  }

  static std::vector<jlm::rvsdg::port>
  CreateOutPorts(const jlm::rvsdg::valuetype & valuetype, size_t numStates)
  {
    std::vector<jlm::rvsdg::port> ports(1, { valuetype });
    std::vector<jlm::rvsdg::port> states(numStates, { llvm::MemoryStateType::Create() });
    ports.insert(ports.end(), states.begin(), states.end());
    ports.emplace_back(jlm::rvsdg::bit64); // addr
    return ports;
  }

  std::string
  debug_string() const override
  {
    return "HLS_LOCAL_LOAD_" + argument(narguments() - 1).type().debug_string();
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new local_load_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(
      jlm::rvsdg::output & index,
      const std::vector<jlm::rvsdg::output *> & states,
      jlm::rvsdg::output & load_result)
  {
    auto region = index.region();
    auto valuetype = dynamic_cast<const jlm::rvsdg::valuetype *>(&load_result.type());
    local_load_op op(*valuetype, states.size());
    std::vector<jlm::rvsdg::output *> inputs;
    inputs.push_back(&index);
    inputs.insert(inputs.end(), states.begin(), states.end());
    inputs.push_back(&load_result);
    return jlm::rvsdg::simple_node::create_normalized(region, op, inputs);
  }
};

class local_store_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~local_store_op()
  {}

  local_store_op(const jlm::rvsdg::valuetype & valuetype, size_t numStates)
      : simple_op(CreateInPorts(valuetype, numStates), CreateOutPorts(valuetype, numStates))
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const local_store_op *>(&other);
    // check predicate and value
    return ot && ot->argument(1).type() == argument(1).type() && ot->narguments() == narguments();
  }

  static std::vector<jlm::rvsdg::port>
  CreateInPorts(const jlm::rvsdg::valuetype & valuetype, size_t numStates)
  {
    std::vector<jlm::rvsdg::port> ports({ jlm::rvsdg::bit64, valuetype });
    std::vector<jlm::rvsdg::port> states(numStates, { llvm::MemoryStateType::Create() });
    ports.insert(ports.end(), states.begin(), states.end());
    return ports;
  }

  static std::vector<jlm::rvsdg::port>
  CreateOutPorts(const jlm::rvsdg::valuetype & valuetype, size_t numStates)
  {
    std::vector<jlm::rvsdg::port> ports(numStates, { llvm::MemoryStateType::Create() });
    ports.emplace_back(jlm::rvsdg::bit64); // addr
    ports.emplace_back(valuetype);   // data
    return ports;
  }

  std::string
  debug_string() const override
  {
    return "HLS_LOCAL_STORE_" + argument(narguments() - 1).type().debug_string();
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new local_store_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(jlm::rvsdg::output & index, jlm::rvsdg::output & value, const std::vector<jlm::rvsdg::output *> & states)
  {
    auto region = index.region();
    auto valuetype = dynamic_cast<const jlm::rvsdg::valuetype *>(&value.type());
    local_store_op op(*valuetype, states.size());
    std::vector<jlm::rvsdg::output *> inputs;
    inputs.push_back(&index);
    inputs.push_back(&value);
    inputs.insert(inputs.end(), states.begin(), states.end());
    return jlm::rvsdg::simple_node::create_normalized(region, op, inputs);
  }
};

class local_mem_req_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~local_mem_req_op()
  {}

  local_mem_req_op(const jlm::llvm::arraytype & at, size_t load_cnt, size_t store_cnt)
      : simple_op(CreateInPorts(at, load_cnt, store_cnt), {})
  {}

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    // TODO:
    auto ot = dynamic_cast<const local_mem_req_op *>(&other);
    // check predicate and value
    return ot && ot->narguments() == narguments()
        && (ot->narguments() == 0 || (ot->argument(1).type() == argument(1).type()))
        && ot->narguments() == narguments();
  }

  static std::vector<jlm::rvsdg::port>
  CreateInPorts(const jlm::llvm::arraytype & at, size_t load_cnt, size_t store_cnt)
  {
    std::vector<jlm::rvsdg::port> ports(1, at);
    for (size_t i = 0; i < load_cnt; ++i)
    {
      ports.emplace_back(jlm::rvsdg::bit64); // addr
    }
    for (size_t i = 0; i < store_cnt; ++i)
    {
      ports.emplace_back(jlm::rvsdg::bit64);       // addr
      ports.emplace_back(at.element_type()); // data
    }
    return ports;
  }

  std::string
  debug_string() const override
  {
    return "HLS_LOCAL_MEM_REQ";
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new local_mem_req_op(*this));
  }

  static std::vector<jlm::rvsdg::output *>
  create(
      jlm::rvsdg::output & mem,
      const std::vector<jlm::rvsdg::output *> & load_operands,
      const std::vector<jlm::rvsdg::output *> & store_operands)
  {
    auto region = mem.region();
    auto at = dynamic_cast<const jlm::llvm::arraytype *>(&mem.type());
    JLM_ASSERT(store_operands.size() % 2 == 0);
    local_mem_req_op op(*at, load_operands.size(), store_operands.size() / 2);
    std::vector<jlm::rvsdg::output *> operands(1, &mem);
    operands.insert(operands.end(), load_operands.begin(), load_operands.end());
    operands.insert(operands.end(), store_operands.begin(), store_operands.end());
    return jlm::rvsdg::simple_node::create_normalized(region, op, operands);
  }
};

}
#endif // JLM_HLS_IR_HLS_HPP

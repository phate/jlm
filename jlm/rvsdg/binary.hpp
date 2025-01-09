/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_BINARY_HPP
#define JLM_RVSDG_BINARY_HPP

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/operation.hpp>
#include <jlm/rvsdg/simple-normal-form.hpp>
#include <jlm/util/common.hpp>

#include <optional>

namespace jlm::rvsdg
{

typedef size_t binop_reduction_path_t;

class BinaryOperation;

class binary_normal_form final : public simple_normal_form
{
public:
  virtual ~binary_normal_form() noexcept;

  binary_normal_form(
      const std::type_info & operator_class,
      jlm::rvsdg::node_normal_form * parent,
      Graph * graph);

  virtual bool
  normalize_node(Node * node) const override;

  virtual std::vector<jlm::rvsdg::output *>
  normalized_create(
      rvsdg::Region * region,
      const SimpleOperation & op,
      const std::vector<jlm::rvsdg::output *> & arguments) const override;

  virtual void
  set_reducible(bool enable);

  inline bool
  get_reducible() const noexcept
  {
    return enable_reducible_;
  }

  virtual void
  set_flatten(bool enable);

  inline bool
  get_flatten() const noexcept
  {
    return enable_flatten_;
  }

  virtual void
  set_reorder(bool enable);

  inline bool
  get_reorder() const noexcept
  {
    return enable_reorder_;
  }

  virtual void
  set_distribute(bool enable);

  inline bool
  get_distribute() const noexcept
  {
    return enable_distribute_;
  }

  virtual void
  set_factorize(bool enable);

  inline bool
  get_factorize() const noexcept
  {
    return enable_factorize_;
  }

private:
  bool
  normalize_node(Node * node, const BinaryOperation & op) const;

  bool enable_reducible_;
  bool enable_reorder_;
  bool enable_flatten_;
  bool enable_distribute_;
  bool enable_factorize_;

  friend class flattened_binary_normal_form;
};

class flattened_binary_normal_form final : public simple_normal_form
{
public:
  virtual ~flattened_binary_normal_form() noexcept;

  flattened_binary_normal_form(
      const std::type_info & operator_class,
      jlm::rvsdg::node_normal_form * parent,
      Graph * graph);

  virtual bool
  normalize_node(Node * node) const override;

  virtual std::vector<jlm::rvsdg::output *>
  normalized_create(
      rvsdg::Region * region,
      const SimpleOperation & op,
      const std::vector<jlm::rvsdg::output *> & arguments) const override;
};

/**
 * Binary operation taking two arguments (with well-defined reduction for more
 * operands if operator is associative).
 */
class BinaryOperation : public SimpleOperation
{
public:
  enum class flags
  {
    none = 0,
    associative = 1,
    commutative = 2
  };

  ~BinaryOperation() noexcept override;

  BinaryOperation(
      const std::vector<std::shared_ptr<const jlm::rvsdg::Type>> operands,
      std::shared_ptr<const jlm::rvsdg::Type> result)
      : SimpleOperation(std::move(operands), { std::move(result) })
  {}

  virtual binop_reduction_path_t
  can_reduce_operand_pair(const jlm::rvsdg::output * op1, const jlm::rvsdg::output * op2)
      const noexcept = 0;

  virtual jlm::rvsdg::output *
  reduce_operand_pair(
      binop_reduction_path_t path,
      jlm::rvsdg::output * op1,
      jlm::rvsdg::output * op2) const = 0;

  virtual BinaryOperation::flags
  flags() const noexcept;

  inline bool
  is_associative() const noexcept;

  inline bool
  is_commutative() const noexcept;

  static jlm::rvsdg::binary_normal_form *
  normal_form(Graph * graph) noexcept
  {
    return static_cast<jlm::rvsdg::binary_normal_form *>(
        graph->GetNodeNormalForm(typeid(BinaryOperation)));
  }
};

/**
 * \brief Flattens a cascade of the same binary operations into a single flattened binary operation.
 *
 * o1 = binaryNode i1 i2
 * o2 = binaryNode o1 i3
 * =>
 * o2 = flattenedBinaryNode i1 i2 i3
 *
 * \pre The binary operation must be associative.
 *
 * @param operation The binary operation on which the transformation is performed.
 * @param operands The operands of the binary node.
 * @return If the normalization could be applied, then the results of the binary operation after
 * the transformation. Otherwise, std::nullopt.
 */
std::optional<std::vector<rvsdg::output *>>
FlattenAssociativeBinaryOperation(
    const BinaryOperation & operation,
    const std::vector<rvsdg::output *> & operands);

/**
 * \brief Applies the reductions implemented in the binary operations reduction functions.
 *
 * @param operation The binary operation on which the transformation is performed.
 * @param operands The operands of the binary node.
 *
 * @return If the normalization could be applied, then the results of the binary operation after
 * the transformation. Otherwise, std::nullopt.
 *
 * \see binary_op::can_reduce_operand_pair()
 * \see binary_op::reduce_operand_pair()
 */
std::optional<std::vector<rvsdg::output *>>
NormalizeBinaryOperation(
    const BinaryOperation & operation,
    const std::vector<rvsdg::output *> & operands);

class flattened_binary_op final : public SimpleOperation
{
public:
  enum class reduction
  {
    linear,
    parallel
  };

  virtual ~flattened_binary_op() noexcept;

  inline flattened_binary_op(std::unique_ptr<BinaryOperation> op, size_t narguments) noexcept
      : SimpleOperation({ narguments, op->argument(0) }, { op->result(0) }),
        op_(std::move(op))
  {
    JLM_ASSERT(op_->is_associative());
  }

  flattened_binary_op(const BinaryOperation & op, size_t narguments)
      : SimpleOperation({ narguments, op.argument(0) }, { op.result(0) }),
        op_(std::unique_ptr<BinaryOperation>(static_cast<BinaryOperation *>(op.copy().release())))
  {
    JLM_ASSERT(op_->is_associative());
  }

  virtual bool
  operator==(const Operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  const BinaryOperation &
  bin_operation() const noexcept
  {
    return *op_;
  }

  static jlm::rvsdg::flattened_binary_normal_form *
  normal_form(Graph * graph) noexcept
  {
    return static_cast<flattened_binary_normal_form *>(
        graph->GetNodeNormalForm(typeid(flattened_binary_op)));
  }

  jlm::rvsdg::output *
  reduce(
      const flattened_binary_op::reduction & reduction,
      const std::vector<jlm::rvsdg::output *> & operands) const;

  static void
  reduce(rvsdg::Region * region, const flattened_binary_op::reduction & reduction);

  static inline void
  reduce(Graph * graph, const flattened_binary_op::reduction & reduction)
  {
    reduce(&graph->GetRootRegion(), reduction);
  }

private:
  std::unique_ptr<BinaryOperation> op_;
};

/**
 * \brief Applies the reductions of the binary operation represented by the flattened binary
 * operation.
 *
 * @param operation The flattened binary operation on which the transformation is performed.
 * @param operands The operands of the flattened binary node.
 *
 * @return If the normalization could be applied, then the results of the flattened binary operation
 * after the transformation. Otherwise, std::nullopt.
 *
 * \see NormalizeBinaryOperation()
 */
std::optional<std::vector<rvsdg::output *>>
NormalizeFlattenedBinaryOperation(
    const flattened_binary_op & operation,
    const std::vector<rvsdg::output *> & operands);

/* binary flags operators */

static constexpr enum BinaryOperation::flags
operator|(enum BinaryOperation::flags a, enum BinaryOperation::flags b)
{
  return static_cast<enum BinaryOperation::flags>(static_cast<int>(a) | static_cast<int>(b));
}

static constexpr enum BinaryOperation::flags
operator&(enum BinaryOperation::flags a, enum BinaryOperation::flags b)
{
  return static_cast<enum BinaryOperation::flags>(static_cast<int>(a) & static_cast<int>(b));
}

/* binary methods */

inline bool
BinaryOperation::is_associative() const noexcept
{
  return static_cast<int>(flags() & BinaryOperation::flags::associative);
}

inline bool
BinaryOperation::is_commutative() const noexcept
{
  return static_cast<int>(flags() & BinaryOperation::flags::commutative);
}

static const binop_reduction_path_t binop_reduction_none = 0;
/* both operands are constants */
static const binop_reduction_path_t binop_reduction_constants = 1;
/* can merge both operands into single (using some "simpler" operator) */
static const binop_reduction_path_t binop_reduction_merge = 2;
/* part of left operand can be folded into right */
static const binop_reduction_path_t binop_reduction_lfold = 3;
/* part of right operand can be folded into left */
static const binop_reduction_path_t binop_reduction_rfold = 4;
/* left operand is neutral element */
static const binop_reduction_path_t binop_reduction_lneutral = 5;
/* right operand is neutral element */
static const binop_reduction_path_t binop_reduction_rneutral = 6;
/* both operands have common form which can be factored over op */
static const binop_reduction_path_t binop_reduction_factor = 7;

}

#endif

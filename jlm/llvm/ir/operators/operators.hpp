/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_OPERATORS_HPP
#define JLM_LLVM_IR_OPERATORS_OPERATORS_HPP

#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/ConversionOperations.hpp>
#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/binary.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/type.hpp>
#include <jlm/rvsdg/unary.hpp>

#include <llvm/ADT/APFloat.h>
#include <llvm/IR/InstrTypes.h>
#include <stdexcept>

namespace jlm::llvm
{

/**
 * Operation that picks its value based on which node branched to the current basic block.
 * All SsaPhiOperations must be at the top of their basic blocks.
 *
 * Each operand corresponds to an incoming basic block,
 * and the list of incoming nodes must include every predecessor in the cfg exactly once.
 */
class SsaPhiOperation final : public rvsdg::SimpleOperation
{
public:
  ~SsaPhiOperation() noexcept override;

  SsaPhiOperation(
      std::vector<ControlFlowGraphNode *> incomingNodes,
      const std::shared_ptr<const jlm::rvsdg::Type> & type)
      : SimpleOperation({ incomingNodes.size(), type }, { type }),
        IncomingNodes_(std::move(incomingNodes))
  {}

  SsaPhiOperation(const SsaPhiOperation &) = default;

  SsaPhiOperation &
  operator=(const SsaPhiOperation &) = delete;

  SsaPhiOperation &
  operator=(SsaPhiOperation &&) = delete;

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  const std::shared_ptr<const rvsdg::Type> &
  Type() const noexcept
  {
    return result(0);
  }

  ControlFlowGraphNode *
  GetIncomingNode(size_t n) const noexcept
  {
    JLM_ASSERT(n < narguments());
    return IncomingNodes_[n];
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(
      const std::vector<std::pair<const Variable *, ControlFlowGraphNode *>> & arguments,
      std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    std::vector<ControlFlowGraphNode *> basicBlocks;
    std::vector<const Variable *> operands;
    for (const auto & argument : arguments)
    {
      basicBlocks.push_back(argument.second);
      operands.push_back(argument.first);
    }

    auto phi = std::make_unique<SsaPhiOperation>(std::move(basicBlocks), std::move(type));
    return ThreeAddressCode::create(std::move(phi), operands);
  }

private:
  std::vector<ControlFlowGraphNode *> IncomingNodes_;
};

class AssignmentOperation final : public rvsdg::SimpleOperation
{
public:
  ~AssignmentOperation() noexcept override;

  explicit AssignmentOperation(const std::shared_ptr<const rvsdg::Type> & type)
      : SimpleOperation({ type, type }, {})
  {}

  AssignmentOperation(const AssignmentOperation &) = default;

  AssignmentOperation(AssignmentOperation &&) = default;

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const Variable * rhs, const Variable * lhs)
  {
    if (rhs->type() != lhs->type())
      throw util::Error("LHS and RHS of assignment must have same type.");

    auto operation = std::make_unique<AssignmentOperation>(rhs->Type());
    return ThreeAddressCode::create(std::move(operation), { lhs, rhs });
  }
};

class SelectOperation final : public rvsdg::SimpleOperation
{
public:
  ~SelectOperation() noexcept override;

  explicit SelectOperation(const std::shared_ptr<const rvsdg::Type> & type)
      : SimpleOperation({ rvsdg::BitType::Create(1), type, type }, { type })
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] const jlm::rvsdg::Type &
  type() const noexcept
  {
    return *result(0);
  }

  [[nodiscard]] const std::shared_ptr<const jlm::rvsdg::Type> &
  Type() const noexcept
  {
    return result(0);
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const llvm::Variable * p, const llvm::Variable * t, const llvm::Variable * f)
  {
    auto op = std::make_unique<SelectOperation>(t->Type());
    return ThreeAddressCode::create(std::move(op), { p, t, f });
  }
};

class VectorSelectOperation final : public rvsdg::SimpleOperation
{
public:
  ~VectorSelectOperation() noexcept override;

private:
  VectorSelectOperation(
      const std::shared_ptr<const VectorType> & pt,
      const std::shared_ptr<const VectorType> & vt)
      : SimpleOperation({ pt, vt, vt }, { vt })
  {}

public:
  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] const rvsdg::Type &
  type() const noexcept
  {
    return *result(0);
  }

  [[nodiscard]] const std::shared_ptr<const rvsdg::Type> &
  Type() const noexcept
  {
    return result(0);
  }

  size_t
  size() const noexcept
  {
    return dynamic_cast<const VectorType *>(&type())->size();
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const Variable * p, const Variable * t, const Variable * f)
  {
    if (is<FixedVectorType>(p->type()) && is<FixedVectorType>(t->type()))
      return createVectorSelectTac<FixedVectorType>(p, t, f);

    if (is<ScalableVectorType>(p->type()) && is<ScalableVectorType>(t->type()))
      return createVectorSelectTac<ScalableVectorType>(p, t, f);

    throw util::Error("Expected vector types as operands.");
  }

private:
  template<typename T>
  static std::unique_ptr<ThreeAddressCode>
  createVectorSelectTac(const Variable * p, const Variable * t, const Variable * f)
  {
    auto fvt = static_cast<const T *>(&t->type());
    auto pt = T::Create(jlm::rvsdg::BitType::Create(1), fvt->size());
    auto vt = T::Create(fvt->Type(), fvt->size());
    auto op = std::unique_ptr<VectorSelectOperation>(new VectorSelectOperation(pt, vt));
    return ThreeAddressCode::create(std::move(op), { p, t, f });
  }
};

class BranchOperation final : public rvsdg::SimpleOperation
{
public:
  ~BranchOperation() noexcept override;

  explicit BranchOperation(std::shared_ptr<const rvsdg::ControlType> type)
      : SimpleOperation({ std::move(type) }, {})
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  inline size_t
  nalternatives() const noexcept
  {
    return std::static_pointer_cast<const rvsdg::ControlType>(argument(0))->nalternatives();
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(size_t nalternatives, const Variable * operand)
  {
    auto op = std::make_unique<BranchOperation>(rvsdg::ControlType::Create(nalternatives));
    return ThreeAddressCode::create(std::move(op), { operand });
  }
};

/** \brief ConstantPointerNullOperation class
 *
 * This operator is the Jlm equivalent of LLVM's ConstantPointerNull constant.
 */
class ConstantPointerNullOperation final : public rvsdg::SimpleOperation
{
public:
  ~ConstantPointerNullOperation() noexcept override;

  explicit ConstantPointerNullOperation()
      : SimpleOperation({}, { PointerType::Create() })
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::unique_ptr<ThreeAddressCode>
  createTac()
  {
    return ThreeAddressCode::create(std::make_unique<ConstantPointerNullOperation>(), {});
  }

  static rvsdg::Node &
  createNode(rvsdg::Region & region)
  {
    return rvsdg::CreateOpNode<ConstantPointerNullOperation>(region);
  }
};

/**
 * This operator is the Jlm equivalent of LLVM's ConstantDataArray constant.
 */
class ConstantDataArrayOperation final : public rvsdg::SimpleOperation
{
public:
  ~ConstantDataArrayOperation() noexcept override;

  ConstantDataArrayOperation(const std::shared_ptr<const rvsdg::Type> & type, size_t size)
      : SimpleOperation({ size, type }, { ArrayType::Create(type, size) })
  {
    if (size == 0)
      throw util::Error("size equals zero.");
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  size_t
  size() const noexcept
  {
    return std::static_pointer_cast<const ArrayType>(result(0))->nelements();
  }

  const rvsdg::Type &
  type() const noexcept
  {
    return std::static_pointer_cast<const ArrayType>(result(0))->element_type();
  }

  static std::unique_ptr<ThreeAddressCode>
  create(const std::vector<const Variable *> & elements)
  {
    if (elements.size() == 0)
      throw util::Error("expected at least one element.");

    auto vt = elements[0]->Type();
    if (vt->Kind() != rvsdg::TypeKind::Value)
      throw util::Error("expected value type.");

    auto op = std::make_unique<ConstantDataArrayOperation>(std::move(vt), elements.size());
    return ThreeAddressCode::create(std::move(op), elements);
  }

  static rvsdg::Output *
  Create(const std::vector<rvsdg::Output *> & elements)
  {
    if (elements.empty())
      throw util::Error("Expected at least one element.");

    auto valueType = elements[0]->Type();
    if (valueType->Kind() != rvsdg::TypeKind::Value)
    {
      throw util::Error("Expected value type.");
    }

    return rvsdg::CreateOpNode<ConstantDataArrayOperation>(
               elements,
               std::move(valueType),
               elements.size())
        .output(0);
  }
};

/**
 * The set of possible types of predicates for integer and pointer comparisons.
 * Based on the integer comparison predicates defined in "::llvm::CmpInst".
 */
enum class ICmpPredicate
{
  Eq,
  Ne,
  Ugt,
  Uge,
  Ult,
  Ule,
  Sgt,
  Sge,
  Slt,
  Sle
};

/**
 * Converts the given comparison \p predicate from an LLVM to the corresponding Jlm enum.
 * @throws if the predicate is not an integer comparison predicate
 */
[[nodiscard]] ICmpPredicate
convertICmpPredicateToJlm(::llvm::CmpInst::Predicate predicate);

/**
 * Converts the given comparison \p predicate from an LLVM to the corresponding Jlm enum.
 * @throws if the predicate is not a valid enum value
 */
[[nodiscard]] ::llvm::CmpInst::Predicate
convertICmpPredicateToLlvm(ICmpPredicate predicate);

/**
 * Converts the given comparison \p predicate to a string
 * @throws if the predicate is not a valid enum value
 */
[[nodiscard]] std::string_view
iCmpPredicateToString(ICmpPredicate predicate);

class PtrCmpOperation final : public rvsdg::BinaryOperation
{
public:
  ~PtrCmpOperation() noexcept override;

  PtrCmpOperation(const std::shared_ptr<const PointerType> & ptype, ICmpPredicate predicate)
      : BinaryOperation({ ptype, ptype }, jlm::rvsdg::BitType::Create(1)),
        predicate_(predicate)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  jlm::rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const jlm::rvsdg::Output * op1, const jlm::rvsdg::Output * op2)
      const noexcept override;

  jlm::rvsdg::Output *
  reduce_operand_pair(
      jlm::rvsdg::binop_reduction_path_t path,
      jlm::rvsdg::Output * op1,
      jlm::rvsdg::Output * op2) const override;

  ICmpPredicate
  predicate() const noexcept
  {
    return predicate_;
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(ICmpPredicate predicateKind, const Variable * op1, const Variable * op2)
  {
    auto pt = std::dynamic_pointer_cast<const PointerType>(op1->Type());
    if (!pt)
      throw util::Error("expected pointer type.");

    auto op = std::make_unique<PtrCmpOperation>(std::move(pt), predicateKind);
    return ThreeAddressCode::create(std::move(op), { op1, op2 });
  }

private:
  ICmpPredicate predicate_;
};

/* floating point constant operator */

class ConstantFP final : public rvsdg::SimpleOperation
{
public:
  ~ConstantFP() noexcept override;

  inline ConstantFP(const fpsize & size, const ::llvm::APFloat & constant)
      : SimpleOperation({}, { FloatingPointType::Create(size) }),
        constant_(constant)
  {}

  ConstantFP(std::shared_ptr<const FloatingPointType> fpt, const ::llvm::APFloat & constant)
      : SimpleOperation({}, { std::move(fpt) }),
        constant_(constant)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  inline const ::llvm::APFloat &
  constant() const noexcept
  {
    return constant_;
  }

  inline const fpsize &
  size() const noexcept
  {
    return std::static_pointer_cast<const FloatingPointType>(result(0))->size();
  }

  [[nodiscard]] static std::unique_ptr<ConstantFP>
  create(const ::llvm::APFloat & constant, const std::shared_ptr<const jlm::rvsdg::Type> & type)
  {
    auto ft = std::dynamic_pointer_cast<const FloatingPointType>(type);
    if (!ft)
      throw util::Error("expected floating point type.");

    return std::make_unique<ConstantFP>(std::move(ft), constant);
  }

  [[nodiscard]] static std::unique_ptr<llvm::ThreeAddressCode>
  createTac(const ::llvm::APFloat & constant, const std::shared_ptr<const jlm::rvsdg::Type> & type)
  {
    return ThreeAddressCode::create(create(constant, type), {});
  }

  [[nodiscard]] static rvsdg::Node &
  createNode(rvsdg::Region & region, fpsize size, const ::llvm::APFloat & constant)
  {
    return rvsdg::CreateOpNode<ConstantFP>(region, size, constant);
  }

  /**
   * Helper function for creating a floating point constant value 0 of the correct type.
   */
  [[nodiscard]] static ::llvm::APFloat
  getZeroRepresentation(fpsize size)
  {
    switch (size)
    {
    case fpsize::half:
      return ::llvm::APFloat::getZero(::llvm::APFloat::IEEEhalf());
    case fpsize::flt:
      return ::llvm::APFloat::getZero(::llvm::APFloat::IEEEsingle());
    case fpsize::dbl:
      return ::llvm::APFloat::getZero(::llvm::APFloat::IEEEdouble());
    case fpsize::x86fp80:
      return ::llvm::APFloat::getZero(::llvm::APFloat::x87DoubleExtended());
    case fpsize::fp128:
      return ::llvm::APFloat::getZero(::llvm::APFloat::IEEEquad());
    default:
      JLM_UNREACHABLE("Unknown float size");
    }
  }

private:
  /* FIXME: I would not like to use the APFloat here,
     but I don't have a replacement right now. */
  ::llvm::APFloat constant_;
};

/* floating point comparison operator */

enum class fpcmp
{
  TRUE,
  FALSE,
  oeq,
  ogt,
  oge,
  olt,
  ole,
  one,
  ord,
  ueq,
  ugt,
  uge,
  ult,
  ule,
  une,
  uno
};

class FCmpOperation final : public rvsdg::BinaryOperation
{
public:
  ~FCmpOperation() noexcept override;

  FCmpOperation(const fpcmp & cmp, const fpsize & size)
      : BinaryOperation(
            { FloatingPointType::Create(size), FloatingPointType::Create(size) },
            jlm::rvsdg::BitType::Create(1)),
        cmp_(cmp)
  {}

  FCmpOperation(const fpcmp & cmp, const std::shared_ptr<const FloatingPointType> & fpt)
      : BinaryOperation({ fpt, fpt }, jlm::rvsdg::BitType::Create(1)),
        cmp_(cmp)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  jlm::rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const jlm::rvsdg::Output * op1, const jlm::rvsdg::Output * op2)
      const noexcept override;

  jlm::rvsdg::Output *
  reduce_operand_pair(
      jlm::rvsdg::binop_reduction_path_t path,
      jlm::rvsdg::Output * op1,
      jlm::rvsdg::Output * op2) const override;

  inline const fpcmp &
  cmp() const noexcept
  {
    return cmp_;
  }

  inline const fpsize &
  size() const noexcept
  {
    return std::static_pointer_cast<const FloatingPointType>(argument(0))->size();
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const fpcmp & cmp, const Variable * op1, const Variable * op2)
  {
    auto ft = std::dynamic_pointer_cast<const FloatingPointType>(op1->Type());
    if (!ft)
      throw util::Error("expected floating point type.");

    auto op = std::make_unique<FCmpOperation>(cmp, std::move(ft));
    return ThreeAddressCode::create(std::move(op), { op1, op2 });
  }

private:
  fpcmp cmp_;
};

/** \brief UndefValueOperation class
 *
 * This operator is the Jlm equivalent of LLVM's UndefValue constant.
 */
class UndefValueOperation final : public rvsdg::SimpleOperation
{
public:
  ~UndefValueOperation() noexcept override;

  explicit UndefValueOperation(std::shared_ptr<const jlm::rvsdg::Type> type)
      : SimpleOperation({}, { std::move(type) })
  {}

  UndefValueOperation(const UndefValueOperation &) = default;

  UndefValueOperation &
  operator=(const UndefValueOperation &) = delete;

  UndefValueOperation &
  operator=(UndefValueOperation &&) = delete;

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] const rvsdg::Type &
  GetType() const noexcept
  {
    return *result(0);
  }

  static jlm::rvsdg::Output *
  Create(rvsdg::Region & region, std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    return rvsdg::CreateOpNode<UndefValueOperation>(region, std::move(type)).output(0);
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  Create(std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    auto operation = std::make_unique<UndefValueOperation>(std::move(type));
    return ThreeAddressCode::create(std::move(operation), {});
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  Create(std::shared_ptr<const jlm::rvsdg::Type> type, const std::string & name)
  {
    auto operation = std::make_unique<UndefValueOperation>(std::move(type));
    return ThreeAddressCode::create(std::move(operation), {}, { name });
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  Create(std::unique_ptr<ThreeAddressCodeVariable> result)
  {
    auto & type = result->Type();

    std::vector<std::unique_ptr<ThreeAddressCodeVariable>> results;
    results.push_back(std::move(result));

    auto operation = std::make_unique<UndefValueOperation>(type);
    return ThreeAddressCode::create(std::move(operation), {}, std::move(results));
  }
};

/** \brief PoisonValueOperation class
 *
 * This operator is the Jlm equivalent of LLVM's PoisonValue constant.
 */
class PoisonValueOperation final : public rvsdg::SimpleOperation
{
public:
  ~PoisonValueOperation() noexcept override;

  explicit PoisonValueOperation(std::shared_ptr<const jlm::rvsdg::Type> type)
      : SimpleOperation({}, { std::move(type) })
  {}

  PoisonValueOperation(const PoisonValueOperation &) = default;

  PoisonValueOperation(PoisonValueOperation &&) = delete;

  PoisonValueOperation &
  operator=(const PoisonValueOperation &) = delete;

  PoisonValueOperation &
  operator=(PoisonValueOperation &&) = delete;

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  const jlm::rvsdg::Type &
  GetType() const noexcept
  {
    return *result(0).get();
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  Create(const std::shared_ptr<const jlm::rvsdg::Type> & type)
  {
    auto valueType = CheckAndConvertType(type);

    auto operation = std::make_unique<PoisonValueOperation>(std::move(valueType));
    return ThreeAddressCode::create(std::move(operation), {});
  }

  static jlm::rvsdg::Output *
  Create(rvsdg::Region * region, const std::shared_ptr<const jlm::rvsdg::Type> & type)
  {
    auto valueType = CheckAndConvertType(type);

    return rvsdg::CreateOpNode<PoisonValueOperation>(*region, std::move(valueType)).output(0);
  }

private:
  static std::shared_ptr<const jlm::rvsdg::Type>
  CheckAndConvertType(const std::shared_ptr<const jlm::rvsdg::Type> & type)
  {
    if (type->Kind() == rvsdg::TypeKind::Value)
      return type;

    throw util::Error("Expected value type.");
  }
};

/** \brief FreezeOperation class
 *
 * This operator is the Jlm equivalent of LLVM's freeze operation.
 * It converts undef and poison values into arbitrary, but fixed, values.
 * For all other inputs it is a no-op.
 */
class FreezeOperation final : public rvsdg::UnaryOperation
{
public:
  ~FreezeOperation() noexcept override;

  explicit FreezeOperation(std::shared_ptr<const jlm::rvsdg::Type> type)
      : rvsdg::UnaryOperation(type, type)
  {
    if (type->Kind() != rvsdg::TypeKind::Value)
      throw std::runtime_error("FreezeOperation given non-value type");
  }

  bool
  operator==(const Operation & other) const noexcept override;

  rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::Output * arg) const noexcept override;

  jlm::rvsdg::Output *
  reduce_operand(rvsdg::unop_reduction_path_t path, jlm::rvsdg::Output * arg) const override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  const jlm::rvsdg::Type &
  getType() const noexcept
  {
    return *result(0).get();
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  createTac(const Variable & operand)
  {
    auto operation = std::make_unique<FreezeOperation>(operand.Type());
    return ThreeAddressCode::create(std::move(operation), { &operand });
  }

  static jlm::rvsdg::Node &
  createNode(jlm::rvsdg::Output & operand)
  {
    return rvsdg::CreateOpNode<FreezeOperation>({ &operand }, operand.Type());
  }
};

/* floating point arithmetic operator */

enum class fpop
{
  add,
  sub,
  mul,
  div,
  mod
};

class FBinaryOperation final : public rvsdg::BinaryOperation
{
public:
  ~FBinaryOperation() noexcept override;

  FBinaryOperation(const llvm::fpop & op, const fpsize & size)
      : BinaryOperation(
            { FloatingPointType::Create(size), FloatingPointType::Create(size) },
            FloatingPointType::Create(size)),
        op_(op)
  {}

  FBinaryOperation(const llvm::fpop & op, const std::shared_ptr<const FloatingPointType> & fpt)
      : BinaryOperation({ fpt, fpt }, fpt),
        op_(op)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  jlm::rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const jlm::rvsdg::Output * op1, const jlm::rvsdg::Output * op2)
      const noexcept override;

  jlm::rvsdg::Output *
  reduce_operand_pair(
      jlm::rvsdg::binop_reduction_path_t path,
      jlm::rvsdg::Output * op1,
      jlm::rvsdg::Output * op2) const override;

  inline const llvm::fpop &
  fpop() const noexcept
  {
    return op_;
  }

  inline const fpsize &
  size() const noexcept
  {
    return std::static_pointer_cast<const FloatingPointType>(result(0))->size();
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const llvm::fpop & fpop, const Variable * op1, const Variable * op2)
  {
    auto ft = std::dynamic_pointer_cast<const FloatingPointType>(op1->Type());
    if (!ft)
      throw util::Error("expected floating point type.");

    auto op = std::make_unique<FBinaryOperation>(fpop, ft);
    return ThreeAddressCode::create(std::move(op), { op1, op2 });
  }

private:
  llvm::fpop op_;
};

class FNegOperation final : public rvsdg::UnaryOperation
{
public:
  ~FNegOperation() noexcept override;

  explicit FNegOperation(const fpsize & size)
      : UnaryOperation(FloatingPointType::Create(size), FloatingPointType::Create(size))
  {}

  explicit FNegOperation(const std::shared_ptr<const FloatingPointType> & fpt)
      : UnaryOperation(fpt, fpt)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::Output * output) const noexcept override;

  jlm::rvsdg::Output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::Output * output)
      const override;

  const fpsize &
  size() const noexcept
  {
    return std::static_pointer_cast<const FloatingPointType>(argument(0))->size();
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const Variable * operand)
  {
    auto type = std::dynamic_pointer_cast<const FloatingPointType>(operand->Type());
    if (!type)
      throw util::Error("expected floating point type.");

    auto op = std::make_unique<FNegOperation>(std::move(type));
    return ThreeAddressCode::create(std::move(op), { operand });
  }
};

class VariadicArgumentListOperation final : public rvsdg::SimpleOperation
{
public:
  ~VariadicArgumentListOperation() noexcept override;

  explicit VariadicArgumentListOperation(std::vector<std::shared_ptr<const jlm::rvsdg::Type>> types)
      : SimpleOperation(std::move(types), { VariableArgumentType::Create() })
  {}

  VariadicArgumentListOperation(const VariadicArgumentListOperation &) = default;

  VariadicArgumentListOperation &
  operator=(const VariadicArgumentListOperation &) = delete;

  VariadicArgumentListOperation &
  operator=(VariadicArgumentListOperation &&) = delete;

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const std::vector<const Variable *> & arguments)
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> operands;
    for (const auto & argument : arguments)
      operands.push_back(argument->Type());

    auto op = std::make_unique<VariadicArgumentListOperation>(std::move(operands));
    return ThreeAddressCode::create(std::move(op), arguments);
  }

  static rvsdg::Output *
  Create(rvsdg::Region & region, const std::vector<rvsdg::Output *> & operands)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> operandTypes;
    operandTypes.reserve(operands.size());
    for (auto & operand : operands)
      operandTypes.emplace_back(operand->Type());

    return operands.empty()
             ? rvsdg::CreateOpNode<VariadicArgumentListOperation>(region, std::move(operandTypes))
                   .output(0)
             : rvsdg::CreateOpNode<VariadicArgumentListOperation>(operands, std::move(operandTypes))
                   .output(0);
  }
};

class ConstantStructOperation final : public rvsdg::SimpleOperation
{
public:
  ~ConstantStructOperation() noexcept override;

  explicit ConstantStructOperation(std::shared_ptr<const StructType> type)
      : SimpleOperation(create_srctypes(*type), { type })
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  const StructType &
  type() const noexcept
  {
    return *std::static_pointer_cast<const StructType>(result(0));
  }

  static std::unique_ptr<ThreeAddressCode>
  create(
      const std::vector<const Variable *> & elements,
      const std::shared_ptr<const rvsdg::Type> & type)
  {
    auto structType = CheckAndExtractStructType(type);

    auto op = std::make_unique<ConstantStructOperation>(std::move(structType));
    return ThreeAddressCode::create(std::move(op), elements);
  }

  static rvsdg::Output &
  Create(
      rvsdg::Region &,
      const std::vector<rvsdg::Output *> & operands,
      std::shared_ptr<const rvsdg::Type> resultType)
  {
    auto structType = CheckAndExtractStructType(std::move(resultType));
    return *rvsdg::CreateOpNode<ConstantStructOperation>(operands, std::move(structType)).output(0);
  }

private:
  static std::vector<std::shared_ptr<const rvsdg::Type>>
  create_srctypes(const StructType & type)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types;
    for (size_t n = 0; n < type.numElements(); n++)
      types.push_back(type.getElementType(n));

    return types;
  }

  static std::shared_ptr<const StructType>
  CheckAndExtractStructType(std::shared_ptr<const rvsdg::Type> type)
  {
    if (auto structType = std::dynamic_pointer_cast<const StructType>(type))
    {
      return structType;
    }

    throw util::TypeError("StructType", type->debug_string());
  }
};

class ConstantArrayOperation final : public rvsdg::SimpleOperation
{
public:
  ~ConstantArrayOperation() noexcept override;

  ConstantArrayOperation(const std::shared_ptr<const jlm::rvsdg::Type> & type, size_t size)
      : SimpleOperation({ size, type }, { ArrayType::Create(type, size) })
  {
    if (size == 0)
      throw util::Error("size equals zero.\n");
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  size_t
  size() const noexcept
  {
    return std::static_pointer_cast<const ArrayType>(result(0))->nelements();
  }

  const jlm::rvsdg::Type &
  type() const noexcept
  {
    return std::static_pointer_cast<const ArrayType>(result(0))->element_type();
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const std::vector<const Variable *> & elements)
  {
    if (elements.size() == 0)
      throw util::Error("expected at least one element.\n");

    auto vt = elements[0]->Type();
    if (vt->Kind() != rvsdg::TypeKind::Value)
      throw util::Error("expected value Type.\n");

    auto op = std::make_unique<ConstantArrayOperation>(vt, elements.size());
    return ThreeAddressCode::create(std::move(op), elements);
  }

  static rvsdg::Output *
  Create(const std::vector<rvsdg::Output *> & operands)
  {
    if (operands.empty())
      throw util::Error("Expected at least one element.\n");

    auto valueType = operands[0]->Type();
    if (valueType->Kind() != rvsdg::TypeKind::Value)
    {
      throw util::Error("Expected value type.\n");
    }

    return rvsdg::CreateOpNode<ConstantArrayOperation>(operands, valueType, operands.size())
        .output(0);
  }
};

class ConstantAggregateZeroOperation final : public rvsdg::SimpleOperation
{
public:
  ~ConstantAggregateZeroOperation() noexcept override;

  explicit ConstantAggregateZeroOperation(std::shared_ptr<const rvsdg::Type> type)
      : SimpleOperation({}, { type })
  {
    auto st = dynamic_cast<const StructType *>(type.get());
    auto at = dynamic_cast<const ArrayType *>(type.get());
    auto vt = dynamic_cast<const VectorType *>(type.get());
    if (!st && !at && !vt)
      throw util::Error("expected array, struct, or vector type.\n");
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    auto op = std::make_unique<ConstantAggregateZeroOperation>(std::move(type));
    return ThreeAddressCode::create(std::move(op), {});
  }

  static jlm::rvsdg::Output *
  Create(rvsdg::Region & region, std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    return rvsdg::CreateOpNode<ConstantAggregateZeroOperation>(region, std::move(type)).output(0);
  }
};

class ExtractElementOperation final : public rvsdg::SimpleOperation
{
public:
  ~ExtractElementOperation() noexcept override;

  ExtractElementOperation(
      const std::shared_ptr<const VectorType> & vtype,
      const std::shared_ptr<const jlm::rvsdg::BitType> & btype)
      : SimpleOperation({ vtype, btype }, { vtype->Type() })
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static inline std::unique_ptr<llvm::ThreeAddressCode>
  create(const llvm::Variable * vector, const llvm::Variable * index)
  {
    auto vt = std::dynamic_pointer_cast<const VectorType>(vector->Type());
    if (!vt)
      throw util::Error("expected vector type.");

    auto bt = std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(index->Type());
    if (!bt)
      throw util::Error("expected bit type.");

    auto op = std::make_unique<ExtractElementOperation>(vt, bt);
    return ThreeAddressCode::create(std::move(op), { vector, index });
  }
};

class ShuffleVectorOperation final : public rvsdg::SimpleOperation
{
public:
  ~ShuffleVectorOperation() noexcept override;

  ShuffleVectorOperation(
      const std::shared_ptr<const FixedVectorType> & v,
      const std::vector<int> & mask)
      : SimpleOperation({ v, v }, { v }),
        Mask_(mask)
  {}

  ShuffleVectorOperation(
      const std::shared_ptr<const ScalableVectorType> & v,
      const std::vector<int> & mask)
      : SimpleOperation({ v, v }, { v }),
        Mask_(mask)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  const ::llvm::ArrayRef<int>
  Mask() const
  {
    return Mask_;
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const Variable * v1, const Variable * v2, const std::vector<int> & mask)
  {
    if (is<FixedVectorType>(v1->type()) && is<FixedVectorType>(v2->type()))
      return CreateShuffleVectorTac<FixedVectorType>(v1, v2, mask);

    if (is<ScalableVectorType>(v1->type()) && is<ScalableVectorType>(v2->type()))
      return CreateShuffleVectorTac<ScalableVectorType>(v1, v2, mask);

    throw util::Error("Expected vector types as operands.");
  }

private:
  template<typename T>
  static std::unique_ptr<ThreeAddressCode>
  CreateShuffleVectorTac(const Variable * v1, const Variable * v2, const std::vector<int> & mask)
  {
    auto vt = std::static_pointer_cast<const T>(v1->Type());
    auto op = std::make_unique<ShuffleVectorOperation>(vt, mask);
    return ThreeAddressCode::create(std::move(op), { v1, v2 });
  }

  std::vector<int> Mask_;
};

class ConstantVectorOperation final : public rvsdg::SimpleOperation
{
public:
  ~ConstantVectorOperation() noexcept override;

  explicit ConstantVectorOperation(const std::shared_ptr<const VectorType> & vt)
      : SimpleOperation({ vt->size(), vt->Type() }, { vt })
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static inline std::unique_ptr<llvm::ThreeAddressCode>
  create(
      const std::vector<const Variable *> & operands,
      const std::shared_ptr<const jlm::rvsdg::Type> & type)
  {
    auto vt = std::dynamic_pointer_cast<const VectorType>(type);
    if (!vt)
      throw util::Error("expected vector type.");

    auto op = std::make_unique<ConstantVectorOperation>(vt);
    return ThreeAddressCode::create(std::move(op), operands);
  }
};

class InsertElementOperation final : public rvsdg::SimpleOperation
{
public:
  ~InsertElementOperation() noexcept override;

  InsertElementOperation(
      const std::shared_ptr<const VectorType> & vectype,
      const std::shared_ptr<const jlm::rvsdg::Type> & vtype,
      const std::shared_ptr<const jlm::rvsdg::BitType> & btype)
      : SimpleOperation({ vectype, vtype, btype }, { vectype })
  {
    if (vectype->type() != *vtype)
    {
      auto received = vtype->debug_string();
      auto expected = vectype->type().debug_string();
      throw util::Error(jlm::util::strfmt("expected ", expected, ", got ", received));
    }
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static inline std::unique_ptr<llvm::ThreeAddressCode>
  create(const llvm::Variable * vector, const llvm::Variable * value, const llvm::Variable * index)
  {
    auto vct = std::dynamic_pointer_cast<const VectorType>(vector->Type());
    if (!vct)
      throw util::Error("expected vector type.");

    auto vt = value->Type();
    if (vt->Kind() != rvsdg::TypeKind::Value)
      throw util::Error("expected value type.");

    auto bt = std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(index->Type());
    if (!bt)
      throw util::Error("expected bit type.");

    auto op = std::make_unique<InsertElementOperation>(vct, vt, bt);
    return ThreeAddressCode::create(std::move(op), { vector, value, index });
  }
};

class VectorUnaryOperation final : public rvsdg::SimpleOperation
{
public:
  ~VectorUnaryOperation() noexcept override;

  VectorUnaryOperation(
      const rvsdg::UnaryOperation & op,
      const std::shared_ptr<const VectorType> & operand,
      const std::shared_ptr<const VectorType> & result)
      : SimpleOperation({ operand }, { result }),
        op_(op.copy())
  {
    // Bit casts can convert between vectors of different length, or between scalars and vectors,
    // so it should not be seen as a vector operation performed lane-wise.
    JLM_ASSERT(!is<BitCastOperation>(op));

    if (operand->type() != *op.argument(0))
    {
      auto received = operand->type().debug_string();
      auto expected = op.argument(0)->debug_string();
      throw util::Error(jlm::util::strfmt("expected ", expected, ", got ", received));
    }

    if (result->type() != *op.result(0))
    {
      auto received = result->type().debug_string();
      auto expected = op.result(0)->debug_string();
      throw util::Error(jlm::util::strfmt("expected ", expected, ", got ", received));
    }
  }

  VectorUnaryOperation(const VectorUnaryOperation & other)
      : SimpleOperation(other),
        op_(other.op_->copy())
  {}

  VectorUnaryOperation(VectorUnaryOperation && other) noexcept
      : SimpleOperation(other),
        op_(std::move(other.op_))
  {}

  VectorUnaryOperation &
  operator=(const VectorUnaryOperation & other)
  {
    if (this != &other)
      op_ = other.op_->copy();

    return *this;
  }

  VectorUnaryOperation &
  operator=(VectorUnaryOperation && other) noexcept
  {
    if (this != &other)
      op_ = std::move(other.op_);

    return *this;
  }

  const rvsdg::UnaryOperation &
  operation() const noexcept
  {
    return *static_cast<const rvsdg::UnaryOperation *>(op_.get());
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static inline std::unique_ptr<llvm::ThreeAddressCode>
  create(
      const rvsdg::UnaryOperation & unop,
      const llvm::Variable * operand,
      const std::shared_ptr<const jlm::rvsdg::Type> & type)
  {
    auto vct1 = std::dynamic_pointer_cast<const VectorType>(operand->Type());
    auto vct2 = std::dynamic_pointer_cast<const VectorType>(type);
    if (!vct1 || !vct2)
      throw util::Error("expected vector type.");

    auto op = std::make_unique<VectorUnaryOperation>(unop, vct1, vct2);
    return ThreeAddressCode::create(std::move(op), { operand });
  }

private:
  std::unique_ptr<Operation> op_;
};

class VectorBinaryOperation final : public rvsdg::SimpleOperation
{
public:
  ~VectorBinaryOperation() noexcept override;

  VectorBinaryOperation(
      const rvsdg::BinaryOperation & binop,
      const std::shared_ptr<const VectorType> & op1,
      const std::shared_ptr<const VectorType> & op2,
      const std::shared_ptr<const VectorType> & result)
      : SimpleOperation({ op1, op2 }, { result }),
        op_(binop.copy())
  {
    if (*op1 != *op2)
      throw util::Error("expected the same vector types.");

    if (op1->type() != *binop.argument(0))
    {
      auto received = op1->type().debug_string();
      auto expected = binop.argument(0)->debug_string();
      throw util::Error(jlm::util::strfmt("expected ", expected, ", got ", received));
    }

    if (result->type() != *binop.result(0))
    {
      auto received = result->type().debug_string();
      auto expected = binop.result(0)->debug_string();
      throw util::Error(jlm::util::strfmt("expected ", expected, ", got ", received));
    }
  }

  VectorBinaryOperation(const VectorBinaryOperation & other)
      : SimpleOperation(other),
        op_(other.op_->copy())
  {}

  VectorBinaryOperation(VectorBinaryOperation && other) noexcept
      : SimpleOperation(other),
        op_(std::move(other.op_))
  {}

  VectorBinaryOperation &
  operator=(const VectorBinaryOperation & other)
  {
    if (this != &other)
      op_ = other.op_->copy();

    return *this;
  }

  VectorBinaryOperation &
  operator=(VectorBinaryOperation && other) noexcept
  {
    if (this != &other)
      op_ = std::move(other.op_);

    return *this;
  }

  const rvsdg::BinaryOperation &
  operation() const noexcept
  {
    return *static_cast<const rvsdg::BinaryOperation *>(op_.get());
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static inline std::unique_ptr<llvm::ThreeAddressCode>
  create(
      const rvsdg::BinaryOperation & binop,
      const llvm::Variable * op1,
      const llvm::Variable * op2,
      const std::shared_ptr<const jlm::rvsdg::Type> & type)
  {
    auto vct1 = std::dynamic_pointer_cast<const VectorType>(op1->Type());
    auto vct2 = std::dynamic_pointer_cast<const VectorType>(op2->Type());
    auto vct3 = std::dynamic_pointer_cast<const VectorType>(type);
    if (!vct1 || !vct2 || !vct3)
      throw util::Error("expected vector type.");

    auto op = std::make_unique<VectorBinaryOperation>(binop, vct1, vct2, vct3);
    return ThreeAddressCode::create(std::move(op), { op1, op2 });
  }

private:
  std::unique_ptr<Operation> op_;
};

class ConstantDataVectorOperation final : public rvsdg::SimpleOperation
{
public:
  ~ConstantDataVectorOperation() noexcept override;

private:
  explicit ConstantDataVectorOperation(const std::shared_ptr<const VectorType> & vt)
      : SimpleOperation({ vt->size(), vt->Type() }, { vt })
  {}

public:
  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  size_t
  size() const noexcept
  {
    return std::static_pointer_cast<const VectorType>(result(0))->size();
  }

  const jlm::rvsdg::Type &
  type() const noexcept
  {
    return std::static_pointer_cast<const VectorType>(result(0))->type();
  }

  static std::unique_ptr<ThreeAddressCode>
  Create(const std::vector<const Variable *> & elements)
  {
    if (elements.empty())
      throw util::Error("Expected at least one element.");

    auto vt = elements[0]->Type();
    if (vt->Kind() != rvsdg::TypeKind::Value)
      throw util::Error("Expected value type.");

    auto op = std::unique_ptr<ConstantDataVectorOperation>(
        new ConstantDataVectorOperation(FixedVectorType::Create(vt, elements.size())));
    return ThreeAddressCode::create(std::move(op), elements);
  }
};

/**
 * Represents the standard C library call malloc() used for dynamically allocating memory.
 *
 * This operation has no equivalent LLVM instruction.
 */
class MallocOperation final : public rvsdg::SimpleOperation
{
public:
  ~MallocOperation() noexcept override;

  explicit MallocOperation(std::shared_ptr<const rvsdg::BitType> type)
      : SimpleOperation(
            { std::move(type), IOStateType::Create() },
            { PointerType::Create(), IOStateType::Create(), MemoryStateType::Create() })
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  const rvsdg::BitType &
  getSizeType() const noexcept
  {
    return *std::static_pointer_cast<const rvsdg::BitType>(argument(0));
  }

  rvsdg::FunctionType
  getFunctionType() const
  {
    JLM_ASSERT(narguments() == 2 && nresults() == 3);
    return rvsdg::FunctionType({ argument(0), argument(1) }, { result(0), result(1), result(2) });
  }

  static rvsdg::Input &
  sizeInput(const rvsdg::Node & node)
  {
    JLM_ASSERT(is<MallocOperation>(node.GetOperation()));
    auto & size = *node.input(0);
    JLM_ASSERT(is<rvsdg::BitType>(size.Type()));
    return size;
  }

  static rvsdg::Input &
  ioStateInput(const rvsdg::Node & node)
  {
    JLM_ASSERT(is<MallocOperation>(node.GetOperation()));
    auto & ioState = *node.input(1);
    JLM_ASSERT(is<IOStateType>(ioState.Type()));
    return ioState;
  }

  static rvsdg::Output &
  addressOutput(const rvsdg::Node & node)
  {
    JLM_ASSERT(is<MallocOperation>(node.GetOperation()));
    auto & address = *node.output(0);
    JLM_ASSERT(is<PointerType>(address.Type()));
    return address;
  }

  static rvsdg::Output &
  ioStateOutput(const rvsdg::Node & node)
  {
    JLM_ASSERT(is<MallocOperation>(node.GetOperation()));
    auto & ioState = *node.output(1);
    JLM_ASSERT(is<IOStateType>(ioState.Type()));
    return ioState;
  }

  static rvsdg::Output &
  memoryStateOutput(const rvsdg::Node & node)
  {
    JLM_ASSERT(is<MallocOperation>(node.GetOperation()));
    auto & memoryState = *node.output(2);
    JLM_ASSERT(is<MemoryStateType>(memoryState.Type()));
    return memoryState;
  }

  static std::unique_ptr<ThreeAddressCode>
  createTac(const Variable * size, const Variable * ioState)
  {
    auto bitType = checkAndExtractSizeType(size->Type());
    auto op = std::make_unique<MallocOperation>(std::move(bitType));
    return ThreeAddressCode::create(std::move(op), { size, ioState });
  }

  static rvsdg::SimpleNode &
  createNode(rvsdg::Output & size, rvsdg::Output & ioState)
  {
    auto bitType = checkAndExtractSizeType(size.Type());
    return rvsdg::CreateOpNode<MallocOperation>({ &size, &ioState }, std::move(bitType));
  }

private:
  static std::shared_ptr<const rvsdg::BitType>
  checkAndExtractSizeType(const std::shared_ptr<const rvsdg::Type> & type)
  {
    if (auto bitType = std::dynamic_pointer_cast<const rvsdg::BitType>(type))
      return bitType;

    throw std::runtime_error("Expected bits type.");
  }
};

/**
 * Represents the standard C library call free() used for freeing dynamically allocated memory.
 *
 * This operation has no equivalent LLVM instruction.
 */
class FreeOperation final : public rvsdg::SimpleOperation
{
public:
  ~FreeOperation() noexcept override;

  explicit FreeOperation(size_t numMemoryStates)
      : SimpleOperation(CreateOperandTypes(numMemoryStates), CreateResultTypes(numMemoryStates))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  /**
   * @param node a SimpleNode containing a FreeOperation
   * @return the input of \p node that takes the pointer value to be freed.
   */
  [[nodiscard]] static rvsdg::Input &
  addressInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(is<FreeOperation>(&node));
    const auto input = node.input(0);
    JLM_ASSERT(is<PointerType>(input->Type()));
    return *input;
  }

  [[nodiscard]] static rvsdg::Input &
  mapMemoryStateOutputToInput(rvsdg::Output & output) noexcept
  {
    JLM_ASSERT(is<MemoryStateType>(output.Type()));
    auto [freeNode, freeOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<FreeOperation>(output);
    JLM_ASSERT(freeOperation);
    const auto input = freeNode->input(output.index() + 1);
    JLM_ASSERT(is<MemoryStateType>(input->Type()));
    return *input;
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  Create(
      const Variable * pointer,
      const std::vector<const Variable *> & memoryStates,
      const Variable * iOState)
  {
    std::vector<const Variable *> operands;
    operands.push_back(pointer);
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());
    operands.push_back(iOState);

    auto operation = std::make_unique<FreeOperation>(memoryStates.size());
    return ThreeAddressCode::create(std::move(operation), operands);
  }

  static std::vector<jlm::rvsdg::Output *>
  Create(
      jlm::rvsdg::Output * pointer,
      const std::vector<jlm::rvsdg::Output *> & memoryStates,
      jlm::rvsdg::Output * iOState)
  {
    std::vector<jlm::rvsdg::Output *> operands;
    operands.push_back(pointer);
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());
    operands.push_back(iOState);

    return outputs(&rvsdg::CreateOpNode<FreeOperation>(operands, memoryStates.size()));
  }

private:
  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateOperandTypes(size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> memoryStates(
        numMemoryStates,
        MemoryStateType::Create());

    std::vector<std::shared_ptr<const rvsdg::Type>> types({ PointerType::Create() });
    types.insert(types.end(), memoryStates.begin(), memoryStates.end());
    types.emplace_back(IOStateType::Create());

    return types;
  }

  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateResultTypes(size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types(
        numMemoryStates,
        MemoryStateType::Create());
    types.emplace_back(IOStateType::Create());

    return types;
  }
};
}

#endif

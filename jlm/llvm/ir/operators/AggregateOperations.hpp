/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_AGGREGATEOPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_AGGREGATEOPERATIONS_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/ir/variable.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

/**
 * Represents LLVM's extractvalue instruction.
 *
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#extractvalue-instruction)
 * for more details.
 */
class ExtractValueOperation final : public rvsdg::SimpleOperation
{
  typedef std::vector<unsigned>::const_iterator const_iterator;

public:
  ~ExtractValueOperation() noexcept override;

  ExtractValueOperation(
      const std::shared_ptr<const rvsdg::Type> & aggtype,
      const std::vector<unsigned> & indices)
      : SimpleOperation({ aggtype }, { dsttype(aggtype, indices) }),
        indices_(indices)
  {
    if (indices.empty())
      throw util::Error("expected at least one index.");
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  const_iterator
  begin() const
  {
    return indices_.begin();
  }

  const_iterator
  end() const
  {
    return indices_.end();
  }

  const rvsdg::Type &
  type() const noexcept
  {
    return *argument(0);
  }

  static std::unique_ptr<ThreeAddressCode>
  create(const Variable * aggregate, const std::vector<unsigned> & indices)
  {
    auto op = std::make_unique<ExtractValueOperation>(aggregate->Type(), indices);
    return ThreeAddressCode::create(std::move(op), { aggregate });
  }

private:
  static std::vector<std::shared_ptr<const rvsdg::Type>>
  dsttype(const std::shared_ptr<const rvsdg::Type> & aggtype, const std::vector<unsigned> & indices)
  {
    std::shared_ptr<const rvsdg::Type> type = aggtype;
    for (const auto & index : indices)
    {
      if (auto st = std::dynamic_pointer_cast<const StructType>(type))
      {
        if (index >= st->numElements())
          throw util::Error("extractvalue index out of bound.");

        type = st->getElementType(index);
      }
      else if (auto at = std::dynamic_pointer_cast<const ArrayType>(type))
      {
        if (index >= at->nelements())
          throw util::Error("extractvalue index out of bound.");

        type = at->GetElementType();
      }
      else
        throw util::Error("expected struct or array type.");
    }

    return { type };
  }

  std::vector<unsigned> indices_{};
};

/**
 * Represents LLVM's insertvalue instruction.
 *
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#insertvalue-instruction)
 * for more details.
 */
class InsertValueOperation final : public rvsdg::SimpleOperation
{
public:
  ~InsertValueOperation() noexcept override;

private:
  InsertValueOperation(
      const std::shared_ptr<const rvsdg::Type> & aggregateType,
      const std::shared_ptr<const rvsdg::Type> & valueType,
      std::vector<unsigned> indices)
      : SimpleOperation({ aggregateType, valueType }, { aggregateType }),
        indices_(std::move(indices))
  {}

public:
  [[nodiscard]] const std::shared_ptr<const rvsdg::Type> &
  getAggregateType() const noexcept
  {
    return argument(0);
  }

  [[nodiscard]] const std::shared_ptr<const rvsdg::Type> &
  getValueType() const noexcept
  {
    return argument(1);
  }

  [[nodiscard]] const std::vector<unsigned> &
  getIndices() const noexcept
  {
    return indices_;
  }

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::unique_ptr<InsertValueOperation>
  create(
      const std::shared_ptr<const rvsdg::Type> & aggregateType,
      const std::shared_ptr<const rvsdg::Type> & valueType,
      std::vector<unsigned> indices)
  {
    checkOperandTypes(aggregateType, valueType, indices);
    return std::unique_ptr<InsertValueOperation>(
        new InsertValueOperation(aggregateType, valueType, std::move(indices)));
  }

  static rvsdg::SimpleNode &
  createNode(
      rvsdg::Output & aggregateOperand,
      rvsdg::Output & valueOperand,
      std::vector<unsigned> indices)
  {
    auto insertValueOperation =
        create(aggregateOperand.Type(), valueOperand.Type(), std::move(indices));
    return rvsdg::SimpleNode::Create(
        *aggregateOperand.region(),
        std::move(insertValueOperation),
        { &aggregateOperand, &valueOperand });
  }

  static std::unique_ptr<ThreeAddressCode>
  createTac(
      const Variable & aggregateOperand,
      const Variable & valueOperand,
      std::vector<unsigned> indices)
  {
    auto insertValueOperation =
        create(aggregateOperand.Type(), valueOperand.Type(), std::move(indices));
    return ThreeAddressCode::create(
        std::move(insertValueOperation),
        { &aggregateOperand, &valueOperand });
  }

private:
  static void
  checkOperandTypes(
      const std::shared_ptr<const rvsdg::Type> & aggregateType,
      const std::shared_ptr<const rvsdg::Type> & valueType,
      const std::vector<unsigned> & indices)
  {
    if (indices.empty())
    {
      throw std::runtime_error("indices are empty.");
    }

    auto type = aggregateType;
    for (const auto & index : indices)
    {
      if (const auto structType = std::dynamic_pointer_cast<const StructType>(type))
      {
        if (index >= structType->numElements())
          throw util::Error("insertvalue index out of bound.");

        type = structType->getElementType(index);
      }
      else if (const auto arrayType = std::dynamic_pointer_cast<const ArrayType>(type))
      {
        if (index >= arrayType->nelements())
          throw util::Error("insertvalue index out of bound.");

        type = arrayType->GetElementType();
      }
      else
      {
        throw std::runtime_error("expected array or struct type.");
      }
    }

    if (*valueType != *type)
    {
      throw std::runtime_error("value operand does not have the right type.");
    }
  }

  std::vector<unsigned> indices_{};
};

}

#endif

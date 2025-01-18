/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_GETELEMENTPTR_HPP
#define JLM_LLVM_IR_OPERATORS_GETELEMENTPTR_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

/**
 * This operation is the equivalent of LLVM's getelementptr instruction.
 *
 * FIXME: We currently do not support vector of pointers for the baseAddress.
 *
 */
class GetElementPtrOperation final : public rvsdg::SimpleOperation
{
public:
  ~GetElementPtrOperation() noexcept override;

public:
  GetElementPtrOperation(
      const std::vector<std::shared_ptr<const rvsdg::bittype>> & offsetTypes,
      std::shared_ptr<const rvsdg::ValueType> pointeeType)
      : SimpleOperation(CreateOperandTypes(offsetTypes), { PointerType::Create() }),
        PointeeType_(std::move(pointeeType))
  {}

  GetElementPtrOperation(const GetElementPtrOperation & other) = default;

  GetElementPtrOperation(GetElementPtrOperation && other) noexcept = default;

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] const rvsdg::ValueType &
  GetPointeeType() const noexcept
  {
    return *dynamic_cast<const rvsdg::ValueType *>(PointeeType_.get());
  }

  /**
   * Creates a GetElementPtr three address code.
   *
   * FIXME: We should not explicitly hand in the resultType parameter, but rather compute it from
   * the pointeeType and the offsets. See LLVM's GetElementPtr instruction for reference.
   *
   * @param baseAddress The base address for the pointer calculation.
   * @param offsets The offsets from the base address.
   * @param pointeeType The type the base address points to.
   * @param resultType The result type of the operation.
   *
   * @return A getElementPtr three address code.
   */
  static std::unique_ptr<llvm::tac>
  Create(
      const variable * baseAddress,
      const std::vector<const variable *> & offsets,
      std::shared_ptr<const rvsdg::ValueType> pointeeType,
      std::shared_ptr<const rvsdg::Type> resultType)
  {
    CheckPointerType(baseAddress->type());
    auto offsetTypes = CheckAndExtractOffsetTypes<const variable>(offsets);
    CheckPointerType(*resultType);

    GetElementPtrOperation operation(offsetTypes, std::move(pointeeType));
    std::vector<const variable *> operands(1, baseAddress);
    operands.insert(operands.end(), offsets.begin(), offsets.end());

    return tac::create(operation, operands);
  }

  /**
   * Creates a GetElementPtr RVSDG node.
   *
   * FIXME: We should not explicitly hand in the resultType parameter, but rather compute it from
   * the pointeeType and the offsets. See LLVM's GetElementPtr instruction for reference.
   *
   * @param baseAddress The base address for the pointer calculation.
   * @param offsets The offsets from the base address.
   * @param pointeeType The type the base address points to.
   * @param resultType The result type of the operation.
   *
   * @return The output of the created GetElementPtr RVSDG node.
   */
  static rvsdg::output *
  Create(
      rvsdg::output * baseAddress,
      const std::vector<rvsdg::output *> & offsets,
      std::shared_ptr<const rvsdg::ValueType> pointeeType,
      std::shared_ptr<const rvsdg::Type> resultType)
  {
    CheckPointerType(baseAddress->type());
    auto offsetTypes = CheckAndExtractOffsetTypes<rvsdg::output>(offsets);
    CheckPointerType(*resultType);

    std::vector operands(1, baseAddress);
    operands.insert(operands.end(), offsets.begin(), offsets.end());

    return rvsdg::CreateOpNode<GetElementPtrOperation>(
               operands,
               offsetTypes,
               std::move(pointeeType))
        .output(0);
  }

private:
  static void
  CheckPointerType(const rvsdg::Type & type)
  {
    if (!is<PointerType>(type))
    {
      throw jlm::util::error("Expected pointer type.");
    }
  }

  template<class T>
  static std::vector<std::shared_ptr<const rvsdg::bittype>>
  CheckAndExtractOffsetTypes(const std::vector<T *> & offsets)
  {
    std::vector<std::shared_ptr<const rvsdg::bittype>> offsetTypes;
    for (const auto & offset : offsets)
    {
      if (auto offsetType = std::dynamic_pointer_cast<const rvsdg::bittype>(offset->Type()))
      {
        offsetTypes.emplace_back(std::move(offsetType));
        continue;
      }

      throw jlm::util::error("Expected bitstring type.");
    }

    return offsetTypes;
  }

  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateOperandTypes(const std::vector<std::shared_ptr<const rvsdg::bittype>> & indexTypes)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types({ PointerType::Create() });
    types.insert(types.end(), indexTypes.begin(), indexTypes.end());

    return types;
  }

  std::shared_ptr<const rvsdg::ValueType> PointeeType_;
};

}

#endif

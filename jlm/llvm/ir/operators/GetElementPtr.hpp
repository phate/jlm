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
      const std::vector<std::shared_ptr<const rvsdg::BitType>> & offsetTypes,
      std::shared_ptr<const rvsdg::Type> pointeeType)
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

  [[nodiscard]] const rvsdg::Type &
  GetPointeeType() const noexcept
  {
    return *PointeeType_.get();
  }

  /**
   * Creates a GetElementPtr three address code.
   *
   * @param baseAddress The base address for the pointer calculation.
   * @param offsets The offsets from the base address.
   * @param pointeeType The type the base address points to.
   *
   * @return A getElementPtr three address code.
   */
  static std::unique_ptr<ThreeAddressCode>
  createTAC(
      const Variable * baseAddress,
      const std::vector<const Variable *> & offsets,
      std::shared_ptr<const rvsdg::Type> pointeeType)
  {
    CheckPointerType(baseAddress->type());
    auto offsetTypes = CheckAndExtractOffsetTypes<const Variable>(offsets);

    auto operation = std::make_unique<GetElementPtrOperation>(offsetTypes, std::move(pointeeType));
    std::vector operands(1, baseAddress);
    operands.insert(operands.end(), offsets.begin(), offsets.end());

    return ThreeAddressCode::create(std::move(operation), operands);
  }

  /**
   * Creates a GetElementPtr RVSDG node.
   *
   * @param baseAddress The base address for the pointer calculation.
   * @param offsets The offsets from the base address.
   * @param pointeeType The type the base address points to.
   *
   * @return The output of the created GetElementPtr RVSDG node.
   */
  static rvsdg::Output *
  Create(
      rvsdg::Output * baseAddress,
      const std::vector<rvsdg::Output *> & offsets,
      std::shared_ptr<const rvsdg::Type> pointeeType)
  {
    CheckPointerType(*baseAddress->Type());
    auto offsetTypes = CheckAndExtractOffsetTypes<rvsdg::Output>(offsets);

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
      throw util::Error("Expected pointer type.");
    }
  }

  template<class T>
  static std::vector<std::shared_ptr<const rvsdg::BitType>>
  CheckAndExtractOffsetTypes(const std::vector<T *> & offsets)
  {
    std::vector<std::shared_ptr<const rvsdg::BitType>> offsetTypes;
    for (const auto & offset : offsets)
    {
      if (auto offsetType = std::dynamic_pointer_cast<const rvsdg::BitType>(offset->Type()))
      {
        offsetTypes.emplace_back(std::move(offsetType));
        continue;
      }

      throw util::Error("Expected bitstring type.");
    }

    return offsetTypes;
  }

  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateOperandTypes(const std::vector<std::shared_ptr<const rvsdg::BitType>> & indexTypes)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types({ PointerType::Create() });
    types.insert(types.end(), indexTypes.begin(), indexTypes.end());

    return types;
  }

  std::shared_ptr<const rvsdg::Type> PointeeType_;
};

}

#endif

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
 * Represents LLVM's getelementptr instruction.
 *
 * See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#getelementptr-instruction) for more details.
 *
 * FIXME: We should type check that pointeeType and the number/types of indices fit together.
 *
 */
class GetElementPtrOperation final : public rvsdg::SimpleOperation
{
public:
  ~GetElementPtrOperation() noexcept override;

  GetElementPtrOperation(
      const std::shared_ptr<const rvsdg::Type> & baseAddressType,
      const std::vector<std::shared_ptr<const rvsdg::BitType>> & indexTypes,
      std::shared_ptr<const rvsdg::Type> pointeeType)
      : SimpleOperation(createOperandTypes(baseAddressType, indexTypes), { baseAddressType }),
        pointeeType_(std::move(pointeeType))
  {
    checkBaseAddressType(*baseAddressType);
  }

  GetElementPtrOperation(const GetElementPtrOperation & other) = default;

  GetElementPtrOperation(GetElementPtrOperation && other) noexcept = default;

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] std::shared_ptr<const rvsdg::Type>
  getPointeeType() const noexcept
  {
    return pointeeType_;
  }

  /**
   * Represents a statically known \ref GetElementPtrOperation.
   */
  struct Constant
  {
    /**
     * @return The byte offset applied by the GEP
     */
    [[nodiscard]] int64_t
    getOffsetInBytes() const noexcept;

    std::shared_ptr<const rvsdg::Type> pointeeType;
    std::vector<uint64_t> indices{};
  };

  /**
   * Attempts to return the \ref GetElementPtrOperation as a \ref GetElementPtrOperation::Constant.
   *
   * @param gepNode A \ref GetElementPtrOperation node
   * @return If all indices are statically known, then a \ref
   * GetElementPtrOperation::Constant is returned, otherwise std::nullopt.
   */
  [[nodiscard]] static std::optional<Constant>
  tryGetAsConstant(const rvsdg::SimpleNode & gepNode);

  /**
   * Returns an iterator range to the indices of a \ref GetElementPtrOperation node.
   *
   * \pre \p node is expected to have a \ref GetElementPtrOperation.
   *
   * @param node A \ref GetElementPtrOperation node.
   * @return An iterator range for all the indices.
   */
  [[nodiscard]] static rvsdg::Node::InputConstIteratorRange
  indices(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(GetElementPtrOperation::numIndices(node) != 0);

    const auto firstIndex = node.input(1);
    JLM_ASSERT(is<rvsdg::BitType>(firstIndex->Type()));
    return { rvsdg::Input::ConstIterator(firstIndex), rvsdg::Input::ConstIterator(nullptr) };
  }

  /**
   * \pre \p node must be a \ref GetElementPtrOperation
   *
   * @param node The \ref GetElementPtrOperation node.
   * @return The number of indices of the node.
   */
  [[nodiscard]] static size_t
  numIndices(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(is<GetElementPtrOperation>(node.GetOperation()));
    return node.ninputs() - 1; // Subtract base address
  }

  /**
   * Returns the base address input of a \ref GetElementPtrOperation node.
   *
   * \pre \p node must be a \ref GetElementPtrOperation.
   *
   * @param node The \ref GetElementPtrOperation node.
   * @return The base address on which the address calculation is performed.
   */
  [[nodiscard]] static rvsdg::Input &
  getBaseAddressInput(rvsdg::Node & node)
  {
    JLM_ASSERT(is<GetElementPtrOperation>(node.GetOperation()));
    const auto baseAddress = node.input(0);
    JLM_ASSERT(is<PointerType>(baseAddress->Type()));
    return *baseAddress;
  }

  /**
   * Returns the base address input of a \ref GetElementPtrOperation node.
   *
   * \pre \p node must be a \ref GetElementPtrOperation
   *
   * @param node The \ref GetElementPtrOperation node.
   * @return The base address on which the address calculation is performed.
   */
  [[nodiscard]] static const rvsdg::Input &
  getBaseAddressInput(const rvsdg::Node & node)
  {
    JLM_ASSERT(is<GetElementPtrOperation>(node.GetOperation()));
    const auto baseAddress = node.input(0);
    JLM_ASSERT(is<PointerType>(baseAddress->Type()));
    return *baseAddress;
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
    auto offsetTypes = checkAndExtractIndexTypes<const Variable>(offsets);

    auto operation = std::make_unique<GetElementPtrOperation>(
        baseAddress->Type(),
        offsetTypes,
        std::move(pointeeType));
    std::vector operands(1, baseAddress);
    operands.insert(operands.end(), offsets.begin(), offsets.end());

    return ThreeAddressCode::create(std::move(operation), operands);
  }

  /**
   * Creates a GetElementPtr RVSDG node.
   *
   * @param baseAddress The base address for the pointer calculation.
   * @param indices The offsets from the base address.
   * @param pointeeType The type the base address points to.
   *
   * @return The created GetElementPtr RVSDG node.
   */
  static rvsdg::SimpleNode &
  createNode(
      rvsdg::Output & baseAddress,
      const std::vector<rvsdg::Output *> & indices,
      std::shared_ptr<const rvsdg::Type> pointeeType)
  {
    const auto indicesTypes = checkAndExtractIndexTypes<rvsdg::Output>(indices);

    std::vector operands(1, &baseAddress);
    operands.insert(operands.end(), indices.begin(), indices.end());

    return rvsdg::CreateOpNode<GetElementPtrOperation>(
        operands,
        baseAddress.Type(),
        indicesTypes,
        std::move(pointeeType));
  }

  /**
   * Creates a GetElementPtr RVSDG node.
   *
   * @param baseAddress The base address for the pointer calculation.
   * @param indices The offsets from the base address.
   * @param pointeeType The type the base address points to.
   *
   * @return The output of the created GetElementPtr RVSDG node.
   */
  static rvsdg::Output *
  create(
      rvsdg::Output * baseAddress,
      const std::vector<rvsdg::Output *> & indices,
      std::shared_ptr<const rvsdg::Type> pointeeType)
  {
    return createNode(*baseAddress, indices, std::move(pointeeType)).output(0);
  }

private:
  static void
  checkBaseAddressType(const rvsdg::Type & type)
  {
    if (!is<PointerType>(type) && !isVectorOf<PointerType>(type))
    {
      throw std::logic_error("Expected pointer type.");
    }
  }

  template<class T>
  static std::vector<std::shared_ptr<const rvsdg::BitType>>
  checkAndExtractIndexTypes(const std::vector<T *> & indices)
  {
    std::vector<std::shared_ptr<const rvsdg::BitType>> offsetTypes;
    for (const auto & offset : indices)
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
  createOperandTypes(
      std::shared_ptr<const rvsdg::Type> baseAddressType,
      const std::vector<std::shared_ptr<const rvsdg::BitType>> & indexTypes)
  {
    std::vector types({ std::move(baseAddressType) });
    types.insert(types.end(), indexTypes.begin(), indexTypes.end());

    return types;
  }

  std::shared_ptr<const rvsdg::Type> pointeeType_;
};

}

#endif

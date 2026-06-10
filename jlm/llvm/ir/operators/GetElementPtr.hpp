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
 * FIXME: We currently do not support vector of pointers for the baseAddress.
 *
 * FIXME: We should type check that pointeeType and the number/types of indices fit together.
 *
 */
class GetElementPtrOperation final : public rvsdg::SimpleOperation
{
public:
  ~GetElementPtrOperation() noexcept override;

  GetElementPtrOperation(
      const std::vector<std::shared_ptr<const rvsdg::BitType>> & indexTypes,
      std::shared_ptr<const rvsdg::Type> pointeeType)
      : SimpleOperation(createOperandTypes(indexTypes), { PointerType::Create() }),
        pointeeType_(std::move(pointeeType))
  {}

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
   * Attempts to find the constant indices of a \ref GetElementPtrOperation node.
   *
   * @param node The \ref GetElementPtrOperation node.
   * @return The constant indices if all of them are found to be constant, otherwise std::nullopt.
   */
  static std::optional<std::vector<uint64_t>>
  tryGetConstantIndices(const rvsdg::Node & node) noexcept;

  /**
   * Represents the \ref GetElementPtrOperation pointee type and its indices.
   */
  struct TypeOffset
  {
    std::shared_ptr<const rvsdg::Type> pointeeType;
    std::vector<uint64_t> indices;
  };

  /**
   * Computes the offsets of the \ref GetElementPtrOperation node \p gepNode, iff they can be
   * statically determined.
   *
   * @param gepNode A \ref GetElementPtrOperation node
   * @return If all indices can be statically determined, then a \ref TypeOffset, otherwise
   * std::nullopt.
   */
  [[nodiscard]] static std::optional<TypeOffset>
  getTypeOffset(const rvsdg::SimpleNode & gepNode);

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
   * Calculates the byte offset applied by the GEP, if the offset is static.
   * The offset is the number of bytes needed to satisfy
   *   output ptr = input ptr + offset in bytes
   *
   * @param gepNode the node representing the \ref GetElementPtrOperation
   * @return the offset applied by the GEP, if it is possible to determine at compile time
   */
  [[nodiscard]] static std::optional<int64_t>
  CalculateOffset(const rvsdg::SimpleNode & gepNode);

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
    checkPointerType(baseAddress->type());
    auto offsetTypes = checkAndExtractIndexTypes<const Variable>(offsets);

    auto operation = std::make_unique<GetElementPtrOperation>(offsetTypes, std::move(pointeeType));
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
    checkPointerType(*baseAddress.Type());
    const auto indicesTypes = checkAndExtractIndexTypes<rvsdg::Output>(indices);

    std::vector operands(1, &baseAddress);
    operands.insert(operands.end(), indices.begin(), indices.end());

    return rvsdg::CreateOpNode<GetElementPtrOperation>(
        operands,
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
  checkPointerType(const rvsdg::Type & type)
  {
    if (!is<PointerType>(type))
    {
      throw util::Error("Expected pointer type.");
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
  createOperandTypes(const std::vector<std::shared_ptr<const rvsdg::BitType>> & indexTypes)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types({ PointerType::Create() });
    types.insert(types.end(), indexTypes.begin(), indexTypes.end());

    return types;
  }

  std::shared_ptr<const rvsdg::Type> pointeeType_;
};

}

#endif

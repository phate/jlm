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
 */
class GetElementPtrOperation final : public rvsdg::SimpleOperation
{
public:
  ~GetElementPtrOperation() noexcept override;

private:
  GetElementPtrOperation(
      const std::shared_ptr<const rvsdg::Type> & baseAddressType,
      const std::vector<std::shared_ptr<const rvsdg::Type>> & indexTypes,
      std::shared_ptr<const rvsdg::Type> gepType,
      std::shared_ptr<const rvsdg::Type> resultType)
      : SimpleOperation(createOperandTypes(baseAddressType, indexTypes), { resultType }),
        gepType_(std::move(gepType))
  {}

public:
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
    return gepType_;
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
   * @param gepType The type used for address calculation.
   *
   * @return A getElementPtr three address code.
   */
  static std::unique_ptr<ThreeAddressCode>
  createTAC(
      const Variable * baseAddress,
      const std::vector<const Variable *> & offsets,
      std::shared_ptr<const rvsdg::Type> gepType)
  {
    auto indexTypes = extractIndexTypes<const Variable>(offsets);
    auto operation = createOperation(baseAddress->Type(), indexTypes, std::move(gepType));

    std::vector operands(1, baseAddress);
    operands.insert(operands.end(), offsets.begin(), offsets.end());

    // FIXME: Validate structural integrity of GEP type
    return ThreeAddressCode::create(std::move(operation), operands);
  }

  static std::unique_ptr<GetElementPtrOperation>
  createOperation(
      const std::shared_ptr<const rvsdg::Type> & baseAddressType,
      const std::vector<std::shared_ptr<const rvsdg::Type>> & indexTypes,
      const std::shared_ptr<const rvsdg::Type> & gepType)
  {
    // 1. Validate that the base address is a pointer or vector of pointers
    checkBaseAddressType(*baseAddressType);

    // 2. Validate that the index types are pointers or vector of integers
    checkIndexTypes(indexTypes);

    // FIXME: Validate vector components align such as uniform lane count, etc.

    auto resultType = getResultType(baseAddressType, indexTypes);

    return std::unique_ptr<GetElementPtrOperation>(
        new GetElementPtrOperation(baseAddressType, indexTypes, gepType, std::move(resultType)));
  }

  /**
   * Creates a GetElementPtr RVSDG node.
   *
   * @param baseAddress The base address for the pointer calculation.
   * @param indices The offsets from the base address.
   * @param gepType The type used for address calculation.
   *
   * @return The created GetElementPtr RVSDG node.
   */
  static rvsdg::SimpleNode &
  createNode(
      rvsdg::Output & baseAddress,
      const std::vector<rvsdg::Output *> & indices,
      const std::shared_ptr<const rvsdg::Type> & gepType)
  {
    std::vector operands({ &baseAddress });
    operands.insert(operands.end(), indices.begin(), indices.end());

    auto indexTypes = extractIndexTypes(indices);
    auto gepOperation = createOperation(baseAddress.Type(), indexTypes, gepType);

    // 4. Validate structural integrity of GEP type
    checkIndexedType(gepType, indices);

    return rvsdg::SimpleNode::Create(*baseAddress.region(), std::move(gepOperation), operands);
  }

  /**
   * Creates a GetElementPtr RVSDG node.
   *
   * @param baseAddress The base address for the pointer calculation.
   * @param indices The offsets from the base address.
   * @param gepType The type used for address calculation.
   *
   * @return The output of the created GetElementPtr RVSDG node.
   */
  static rvsdg::Output *
  create(
      rvsdg::Output * baseAddress,
      const std::vector<rvsdg::Output *> & indices,
      std::shared_ptr<const rvsdg::Type> gepType)
  {
    return createNode(*baseAddress, indices, std::move(gepType)).output(0);
  }

private:
  static std::shared_ptr<const rvsdg::Type>
  getIndexedType(
      const std::shared_ptr<const rvsdg::Type> & gepType,
      const std::vector<rvsdg::Output *> & indices);

  static void
  checkIndexedType(
      const std::shared_ptr<const rvsdg::Type> & gepType,
      const std::vector<rvsdg::Output *> & indices)
  {
    const auto indexedType = getIndexedType(gepType, indices);
    if (indexedType == nullptr)
    {
      throw std::logic_error("Invalid GetElementPtrOperation indices for type!");
    }
  }

  static void
  checkBaseAddressType(const rvsdg::Type & type)
  {
    const auto isPointerType = is<PointerType>(type);
    const auto vectorType = dynamic_cast<const VectorType *>(&type);
    const auto isVectorOfPointerType = vectorType && is<PointerType>(vectorType->Type());

    if (!isPointerType && !isVectorOfPointerType)
    {
      throw std::logic_error("Expected pointer type.");
    }
  }

  static void
  checkIndexTypes(const std::vector<std::shared_ptr<const rvsdg::Type>> & indexTypes)
  {
    for (auto & indexType : indexTypes)
    {
      if (!is<rvsdg::BitType>(indexType) && !isVectorOf<rvsdg::BitType>(*indexType))
      {
        throw std::logic_error("Expected bitstring type.");
      }
    }
  }

  static std::shared_ptr<const rvsdg::Type>
  getResultType(
      const std::shared_ptr<const rvsdg::Type> & baseAddressType,
      const std::vector<std::shared_ptr<const rvsdg::Type>> & indexTypes)
  {
    const auto resultType = PointerType::Create();

    // FIXME: Fix vector type such that it can uniformly handle fixed and scalable vector types
    // similar to LLVM
    if (const auto fixedVectorType =
            std::dynamic_pointer_cast<const FixedVectorType>(baseAddressType))
    {
      return FixedVectorType::Create(resultType, fixedVectorType->size());
    }
    if (const auto scalableVectorType =
            std::dynamic_pointer_cast<const ScalableVectorType>(baseAddressType))
    {
      return ScalableVectorType::Create(resultType, scalableVectorType->size());
    }

    for (auto & indexType : indexTypes)
    {
      if (const auto fixedVectorType = std::dynamic_pointer_cast<const FixedVectorType>(indexType))
      {
        return FixedVectorType::Create(resultType, fixedVectorType->size());
      }
      if (const auto scalableVectorType =
              std::dynamic_pointer_cast<const ScalableVectorType>(indexType))
      {
        return ScalableVectorType::Create(resultType, scalableVectorType->size());
      }
    }

    return resultType;
  }

  template<class T>
  static std::vector<std::shared_ptr<const rvsdg::Type>>
  extractIndexTypes(const std::vector<T *> & indices)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> indexTypes;
    for (const auto & index : indices)
    {
      indexTypes.emplace_back(std::move(index->Type()));
    }

    return indexTypes;
  }

  static std::vector<std::shared_ptr<const rvsdg::Type>>
  createOperandTypes(
      std::shared_ptr<const rvsdg::Type> baseAddressType,
      const std::vector<std::shared_ptr<const rvsdg::Type>> & indexTypes)
  {
    std::vector types({ std::move(baseAddressType) });
    types.insert(types.end(), indexTypes.begin(), indexTypes.end());

    return types;
  }

  std::shared_ptr<const rvsdg::Type> gepType_;
};

}

#endif

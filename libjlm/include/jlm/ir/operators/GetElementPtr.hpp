/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_OPERATORS_GETELEMENTPTR_HPP
#define JLM_IR_OPERATORS_GETELEMENTPTR_HPP

#include <jive/types/bitstring/type.hpp>
#include <jive/rvsdg/simple-node.hpp>

#include <jlm/ir/tac.hpp>
#include <jlm/ir/types.hpp>

namespace jlm
{

/**
 * This operation is the equivalent of LLVM's getelementptr instruction.
 *
 * FIXME: We currently do not support vector of pointers for the baseAddress.
 *
 */
class GetElementPtrOperation final : public jive::simple_op
{
public:
  ~GetElementPtrOperation() noexcept override;

private:
  GetElementPtrOperation(
    const PointerType & baseAddressType,
    const std::vector<jive::bittype> & offsetTypes,
    const jive::valuetype & pointeeType,
    const PointerType & resultType)
    : simple_op(CreateOperandPorts(baseAddressType, offsetTypes), {resultType})
    , PointeeType_(pointeeType.copy())
  {}

public:
  GetElementPtrOperation(const GetElementPtrOperation & other)
  : simple_op(other)
  , PointeeType_(other.PointeeType_->copy())
  {}

  GetElementPtrOperation(GetElementPtrOperation && other) noexcept
  : simple_op(other)
  , PointeeType_(std::move(other.PointeeType_))
  {}

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<jive::operation>
  copy() const override;

  [[nodiscard]] const jive::valuetype &
  GetPointeeType() const noexcept
  {
    return *dynamic_cast<const jive::valuetype*>(PointeeType_.get());
  }

  /**
   * Creates a GetElementPtr three address code.
   *
   * FIXME: We should not explicitly hand in the resultType parameter, but rather compute it from the pointeeType and
   * the offsets. See LLVM's GetElementPtr instruction for reference.
   *
   * @param baseAddress The base address for the pointer calculation.
   * @param offsets The offsets from the base address.
   * @param pointeeType The type the base address points to.
   * @param resultType The result type of the operation.
   *
   * @return A getElementPtr three address code.
   */
  static std::unique_ptr<jlm::tac>
  Create(
    const variable * baseAddress,
    const std::vector<const variable*> & offsets,
    const jive::valuetype & pointeeType,
    const jive::type & resultType)
  {
    auto & baseAddressType = CheckAndExtractPointerType(baseAddress->type());
    auto offsetTypes = CheckAndExtractOffsetTypes<const variable>(offsets);
    auto & checkedResultType = CheckAndExtractPointerType(resultType);

    GetElementPtrOperation op(baseAddressType, offsetTypes, pointeeType, checkedResultType);
    std::vector<const variable*> operands(1, baseAddress);
    operands.insert(operands.end(), offsets.begin(), offsets.end());

    return tac::create(op, operands);
  }

  /**
   * Creates a GetElementPtr RVSDG node.
   *
   * FIXME: We should not explicitly hand in the resultType parameter, but rather compute it from the pointeeType and
   * the offsets. See LLVM's GetElementPtr instruction for reference.
   *
   * @param baseAddress The base address for the pointer calculation.
   * @param offsets The offsets from the base address.
   * @param pointeeType The type the base address points to.
   * @param resultType The result type of the operation.
   *
   * @return The output of the created GetElementPtr RVSDG node.
   */
  static jive::output *
  Create(
    jive::output * baseAddress,
    const std::vector<jive::output*> & offsets,
    const jive::valuetype & pointeeType,
    const jive::type & resultType)
  {
    auto & baseAddressType = CheckAndExtractPointerType(baseAddress->type());
    auto offsetTypes = CheckAndExtractOffsetTypes<jive::output>(offsets);
    auto & checkedResultType = CheckAndExtractPointerType(resultType);

    GetElementPtrOperation op(baseAddressType, offsetTypes, pointeeType, checkedResultType);
    std::vector<jive::output*> operands(1, baseAddress);
    operands.insert(operands.end(), offsets.begin(), offsets.end());

    return jive::simple_node::create_normalized(baseAddress->region(), op, operands)[0];
  }

private:
  static const PointerType &
  CheckAndExtractPointerType(const jive::type & type)
  {
    if (auto pointerType = dynamic_cast<const PointerType*>(&type))
    {
      return *pointerType;
    }

    throw error("Expected pointer type.");
  }

  template<class T> static std::vector<jive::bittype>
  CheckAndExtractOffsetTypes(const std::vector<T*> & offsets)
  {
    std::vector<jive::bittype> offsetTypes;
    for (const auto & offset : offsets)
    {
      if (auto offsetType = dynamic_cast<const jive::bittype*>(&offset->type()))
      {
        offsetTypes.emplace_back(*offsetType);
        continue;
      }

      throw error("Expected bitstring type.");
    }

    return offsetTypes;
  }

  static std::vector<jive::port>
  CreateOperandPorts(
    const PointerType & baseAddressType,
    const std::vector<jive::bittype> & indexTypes)
  {
    std::vector<jive::port> ports({baseAddressType});
    for (const auto & type : indexTypes)
    {
      ports.emplace_back(type);
    }

    return ports;
  }

  std::unique_ptr<jive::type> PointeeType_;
};

}

#endif
/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef TESTS_TEST_OPERATION_HPP
#define TESTS_TEST_OPERATION_HPP

#include <jlm/rvsdg/binary.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/operation.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/type.hpp>
#include <jlm/rvsdg/unary.hpp>

namespace jlm::tests
{

class TestUnaryOperation final : public rvsdg::UnaryOperation
{
public:
  ~TestUnaryOperation() noexcept override;

  TestUnaryOperation(
      std::shared_ptr<const rvsdg::Type> srctype,
      std::shared_ptr<const rvsdg::Type> dsttype) noexcept
      : rvsdg::UnaryOperation(std::move(srctype), std::move(dsttype))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  rvsdg::unop_reduction_path_t
  can_reduce_operand(const rvsdg::Output * operand) const noexcept override;

  rvsdg::Output *
  reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::Output * operand) const override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static rvsdg::Node *
  create(
      rvsdg::Region *,
      std::shared_ptr<const rvsdg::Type> srctype,
      rvsdg::Output * operand,
      std::shared_ptr<const rvsdg::Type> dsttype)
  {
    return &rvsdg::CreateOpNode<TestUnaryOperation>(
        { operand },
        std::move(srctype),
        std::move(dsttype));
  }

  static inline rvsdg::Output *
  create_normalized(
      std::shared_ptr<const rvsdg::Type> srctype,
      rvsdg::Output * operand,
      std::shared_ptr<const rvsdg::Type> dsttype)
  {
    return rvsdg::CreateOpNode<TestUnaryOperation>(
               { operand },
               std::move(srctype),
               std::move(dsttype))
        .output(0);
  }
};

class TestBinaryOperation final : public rvsdg::BinaryOperation
{
public:
  ~TestBinaryOperation() noexcept override;

  TestBinaryOperation(
      const std::shared_ptr<const rvsdg::Type> & srctype,
      std::shared_ptr<const rvsdg::Type> dsttype,
      const enum BinaryOperation::flags & flags) noexcept
      : BinaryOperation({ srctype, srctype }, std::move(dsttype)),
        flags_(flags)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::unop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum BinaryOperation::flags
  flags() const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static rvsdg::Node *
  create(
      const std::shared_ptr<const rvsdg::Type> & srctype,
      std::shared_ptr<const rvsdg::Type> dsttype,
      rvsdg::Output * op1,
      rvsdg::Output * op2)
  {
    return &rvsdg::CreateOpNode<TestBinaryOperation>(
        { op1, op2 },
        srctype,
        std::move(dsttype),
        BinaryOperation::flags::none);
  }

  static inline rvsdg::Output *
  create_normalized(
      const std::shared_ptr<const rvsdg::Type> srctype,
      std::shared_ptr<const rvsdg::Type> dsttype,
      rvsdg::Output * op1,
      rvsdg::Output * op2)
  {
    return rvsdg::CreateOpNode<TestBinaryOperation>(
               { op1, op2 },
               srctype,
               std::move(dsttype),
               flags::none)
        .output(0);
  }

private:
  enum BinaryOperation::flags flags_;
};

class TestStructuralOperation final : public rvsdg::StructuralOperation
{
public:
  ~TestStructuralOperation() noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;
};

class TestStructuralNode final : public rvsdg::StructuralNode
{
public:
  ~TestStructuralNode() noexcept override;

private:
  TestStructuralNode(rvsdg::Region * parent, size_t nsubregions)
      : StructuralNode(parent, nsubregions)
  {}

public:
  /**
   * \brief A variable routed in a \ref TestStructuralNode
   */
  struct InputVar
  {
    rvsdg::StructuralInput * input{};
    std::vector<rvsdg::RegionArgument *> argument{};
  };

  /**
   * \brief A variable routed out of a \ref TestStructuralNode
   */
  struct OutputVar
  {
    rvsdg::StructuralOutput * output{};
    std::vector<rvsdg::RegionResult *> result{};
  };

  /**
   * Add an input WITHOUT subregion arguments to a \ref TestStructuralNode.
   *
   * @param origin Value to be routed in.
   * @return The created input variable.
   */
  rvsdg::StructuralInput &
  addInputOnly(rvsdg::Output & origin);

  /**
   * Add an input WITH subregion arguments to a \ref TestStructuralNode.
   *
   * @param origin Value to be routed in.
   * @return Description of input variable.
   */
  InputVar
  addInputWithArguments(rvsdg::Output & origin);

  /**
   * Removes the input with the given \p index,
   * and any subregion arguments associated with it.
   * All such arguments must be dead.
   * @param index the index of the input to remove.
   */
  void
  removeInputAndArguments(size_t index);

  /**
   * Add subregion arguments WITHOUT an input to a \ref TestStructuralNode.
   * @param type The argument type
   * @return Description of input variable.
   */
  InputVar
  addArguments(const std::shared_ptr<const rvsdg::Type> & type);

  /**
   * Add an output WITHOUT subregion results to a \ref TestStructuralNode.
   *
   * @param type The output type
   * @return The created output variable.
   */
  rvsdg::StructuralOutput &
  addOutputOnly(std::shared_ptr<const rvsdg::Type> type);

  /**
   * Add an output WITH subregion results to a \ref TestStructuralNode.
   *
   * @param origins The values to be routed out.
   * @return Description of output variable.
   */
  OutputVar
  addOutputWithResults(const std::vector<rvsdg::Output *> & origins);

  /**
   * Removes the output with the given \p index,
   * and any subregion results associated with it.
   * @param index the index of the output to remove.
   */
  void
  removeOutputAndResults(size_t index);

  /**
   * Add subregion results WITHOUT output to a \ref TestStructuralNode.
   *
   * @param origins The values to be routed out.
   * @return Description of output variable.
   */
  OutputVar
  addResults(const std::vector<rvsdg::Output *> & origins);

  [[nodiscard]] const TestStructuralOperation &
  GetOperation() const noexcept override;

  static TestStructuralNode *
  create(rvsdg::Region * parent, size_t nsubregions)
  {
    return new TestStructuralNode(parent, nsubregions);
  }

  TestStructuralNode *
  copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const override;
};

class TestOperation final : public rvsdg::SimpleOperation
{
public:
  ~TestOperation() noexcept override;

  TestOperation(
      std::vector<std::shared_ptr<const rvsdg::Type>> arguments,
      std::vector<std::shared_ptr<const rvsdg::Type>> results)
      : SimpleOperation(std::move(arguments), std::move(results))
  {}

  TestOperation(const TestOperation &) = default;

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::unique_ptr<TestOperation>
  create(
      std::vector<std::shared_ptr<const rvsdg::Type>> operandTypes,
      std::vector<std::shared_ptr<const rvsdg::Type>> resultTypes)
  {
    return std::make_unique<TestOperation>(std::move(operandTypes), std::move(resultTypes));
  }

  static rvsdg::SimpleNode *
  createNode(
      rvsdg::Region * region,
      const std::vector<rvsdg::Output *> & operands,
      std::vector<std::shared_ptr<const rvsdg::Type>> resultTypes)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> operandTypes;
    for (const auto & operand : operands)
      operandTypes.push_back(operand->Type());

    return createNode(region, operandTypes, operands, resultTypes);
  }

  static rvsdg::SimpleNode *
  createNode(
      rvsdg::Region * region,
      std::vector<std::shared_ptr<const rvsdg::Type>> operandTypes,
      const std::vector<rvsdg::Output *> & operands,
      std::vector<std::shared_ptr<const rvsdg::Type>> resultTypes)
  {
    return operands.empty() ? &rvsdg::CreateOpNode<TestOperation>(
                                  *region,
                                  std::move(operandTypes),
                                  std::move(resultTypes))
                            : &rvsdg::CreateOpNode<TestOperation>(
                                  { operands },
                                  std::move(operandTypes),
                                  std::move(resultTypes));
  }
};

class TestGraphArgument final : public jlm::rvsdg::RegionArgument
{
private:
  TestGraphArgument(
      rvsdg::Region & region,
      rvsdg::StructuralInput * input,
      std::shared_ptr<const jlm::rvsdg::Type> type)
      : jlm::rvsdg::RegionArgument(&region, input, type)
  {}

public:
  TestGraphArgument &
  Copy(rvsdg::Region & region, rvsdg::StructuralInput * input) const override
  {
    return Create(region, input, Type());
  }

  static TestGraphArgument &
  Create(
      rvsdg::Region & region,
      rvsdg::StructuralInput * input,
      std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    auto graphArgument = new TestGraphArgument(region, input, std::move(type));
    region.addArgument(std::unique_ptr<RegionArgument>(graphArgument));
    return *graphArgument;
  }
};

}

#endif

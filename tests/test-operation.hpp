/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef TESTS_TEST_OPERATION_HPP
#define TESTS_TEST_OPERATION_HPP

#include <jlm/rvsdg/binary.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/nullary.hpp>
#include <jlm/rvsdg/operation.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/type.hpp>
#include <jlm/rvsdg/unary.hpp>

#include <jlm/llvm/ir/tac.hpp>

namespace jlm::tests
{

/**
 * Represents an import into the RVSDG of an external entity.
 * It can be used for testing of graph imports.
 */
class GraphImport final : public rvsdg::GraphImport
{
  GraphImport(rvsdg::Graph & graph, std::shared_ptr<const rvsdg::Type> type, std::string name)
      : rvsdg::GraphImport(graph, std::move(type), std::move(name))
  {}

public:
  GraphImport &
  Copy(rvsdg::Region & region, rvsdg::StructuralInput * input) override;

  static GraphImport &
  Create(rvsdg::Graph & graph, std::shared_ptr<const rvsdg::Type> type, std::string name)
  {
    auto graphImport = new GraphImport(graph, std::move(type), std::move(name));
    graph.GetRootRegion().append_argument(graphImport);
    return *graphImport;
  }
};

/**
 * Represents an export from the RVSDG of an internal entity.
 * It can be used for testing of graph exports.
 */
class GraphExport final : public rvsdg::GraphExport
{
  GraphExport(rvsdg::Output & origin, std::string name)
      : rvsdg::GraphExport(origin, std::move(name))
  {}

public:
  GraphExport &
  Copy(rvsdg::Output & origin, rvsdg::StructuralOutput * output) override;

  static GraphExport &
  Create(rvsdg::Output & origin, std::string name)
  {
    auto graphExport = new GraphExport(origin, std::move(name));
    origin.region()->graph()->GetRootRegion().append_result(graphExport);
    return *graphExport;
  }
};

class NullaryOperation final : public rvsdg::NullaryOperation
{
public:
  explicit NullaryOperation(const std::shared_ptr<const jlm::rvsdg::Type> & resultType)
      : rvsdg::NullaryOperation(resultType)
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    const auto nullaryOperation = dynamic_cast<const NullaryOperation *>(&other);
    return nullaryOperation && *result(0) == *nullaryOperation->result(0);
  }

  [[nodiscard]] std::string
  debug_string() const override
  {
    return "NullaryOperation";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<NullaryOperation>(this->result(0));
  }
};

/* unary operation */

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

static inline bool
is_unary_op(const rvsdg::Operation & op) noexcept
{
  return dynamic_cast<const TestUnaryOperation *>(&op);
}

static inline bool
is_unary_node(const rvsdg::Node * node) noexcept
{
  return jlm::rvsdg::is<TestUnaryOperation>(node);
}

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

class StructuralNodeInput;

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
    rvsdg::Input * input{};
    std::vector<rvsdg::Output *> argument{};
  };

  /**
   * \brief A variable routed out of a \ref TestStructuralNode
   */
  struct OutputVar
  {
    rvsdg::Output * output{};
    std::vector<rvsdg::Input *> result{};
  };

  /**
   * Add an input WITHOUT subregion arguments to a \ref TestStructuralNode.
   *
   * @param origin Value to be routed in.
   * @return Description of input variable.
   */
  InputVar
  AddInput(rvsdg::Output & origin);

  /**
   * Add an input WITH subregion arguments to a \ref TestStructuralNode.
   *
   * @param origin Value to be routed in.
   * @return Description of input variable.
   */
  InputVar
  AddInputWithArguments(rvsdg::Output & origin);

  /**
   * Add subregion arguments WITHOUT an input to a \ref TestStructuralNode.
   * @param type The argument type
   * @return Description of input variable
   */
  InputVar
  AddArguments(const std::shared_ptr<const rvsdg::Type> & type);

  /**
   * Add an output WITHOUT subregion results to a \ref TestStructuralNode.
   *
   * @param type The output type
   * @return Description of output variable.
   */
  OutputVar
  AddOutput(std::shared_ptr<const rvsdg::Type> type);

  /**
   * Add an output WITH subregion results to a \ref TestStructuralNode.
   *
   * @param origins The values to be routed out.
   * @return Description of output variable.
   */
  OutputVar
  AddOutputWithResults(const std::vector<rvsdg::Output *> & origins);

  /**
   * Add subregion results WITHOUT output to a \ref TestStructuralNode.
   *
   * @param origins The values to be routed out.
   * @return Description of output variable.
   */
  OutputVar
  AddResults(const std::vector<rvsdg::Output *> & origins);

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

class StructuralNodeInput final : public rvsdg::StructuralInput
{
  friend TestStructuralNode;

public:
  ~StructuralNodeInput() noexcept override;

private:
  StructuralNodeInput(
      TestStructuralNode & node,
      rvsdg::Output & origin,
      std::shared_ptr<const rvsdg::Type> type)
      : StructuralInput(&node, &origin, std::move(type))
  {}
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

  static rvsdg::SimpleNode *
  create(
      rvsdg::Region * region,
      const std::vector<rvsdg::Output *> & operands,
      std::vector<std::shared_ptr<const rvsdg::Type>> result_types)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> operand_types;
    for (const auto & operand : operands)
      operand_types.push_back(operand->Type());

    return Create(region, operand_types, operands, result_types);
  }

  static rvsdg::SimpleNode *
  Create(
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

static inline std::unique_ptr<llvm::ThreeAddressCode>
create_testop_tac(
    const std::vector<const llvm::Variable *> & arguments,
    std::vector<std::shared_ptr<const rvsdg::Type>> result_types)
{
  std::vector<std::shared_ptr<const rvsdg::Type>> argument_types;
  for (const auto & arg : arguments)
    argument_types.push_back(arg->Type());

  TestOperation op(std::move(argument_types), std::move(result_types));
  return llvm::ThreeAddressCode::create(op, arguments);
}

static inline std::vector<rvsdg::Output *>
create_testop(
    rvsdg::Region * region,
    const std::vector<rvsdg::Output *> & operands,
    std::vector<std::shared_ptr<const rvsdg::Type>> result_types)
{
  std::vector<std::shared_ptr<const rvsdg::Type>> operand_types;
  for (const auto & operand : operands)
    operand_types.push_back(operand->Type());

  return operands.empty() ? outputs(&rvsdg::CreateOpNode<TestOperation>(
                                *region,
                                std::move(operand_types),
                                std::move(result_types)))
                          : outputs(&rvsdg::CreateOpNode<TestOperation>(
                                operands,
                                std::move(operand_types),
                                std::move(result_types)));
}

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
  Copy(rvsdg::Region & region, rvsdg::StructuralInput * input) override
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
    region.append_argument(graphArgument);
    return *graphArgument;
  }
};

}

#endif

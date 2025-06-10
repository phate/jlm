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

class unary_op final : public rvsdg::UnaryOperation
{
public:
  virtual ~unary_op() noexcept;

  inline unary_op(
      std::shared_ptr<const rvsdg::Type> srctype,
      std::shared_ptr<const rvsdg::Type> dsttype) noexcept
      : rvsdg::UnaryOperation(std::move(srctype), std::move(dsttype))
  {}

  virtual bool
  operator==(const Operation & other) const noexcept override;

  virtual rvsdg::unop_reduction_path_t
  can_reduce_operand(const rvsdg::Output * operand) const noexcept override;

  virtual rvsdg::Output *
  reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::Output * operand) const override;

  virtual std::string
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
    return &rvsdg::CreateOpNode<unary_op>({ operand }, std::move(srctype), std::move(dsttype));
  }

  static inline rvsdg::Output *
  create_normalized(
      std::shared_ptr<const rvsdg::Type> srctype,
      rvsdg::Output * operand,
      std::shared_ptr<const rvsdg::Type> dsttype)
  {
    return rvsdg::CreateOpNode<unary_op>({ operand }, std::move(srctype), std::move(dsttype))
        .output(0);
  }
};

static inline bool
is_unary_op(const rvsdg::Operation & op) noexcept
{
  return dynamic_cast<const unary_op *>(&op);
}

static inline bool
is_unary_node(const rvsdg::Node * node) noexcept
{
  return jlm::rvsdg::is<unary_op>(node);
}

/* binary operation */

class binary_op final : public rvsdg::BinaryOperation
{
public:
  virtual ~binary_op() noexcept;

  inline binary_op(
      const std::shared_ptr<const rvsdg::Type> & srctype,
      std::shared_ptr<const rvsdg::Type> dsttype,
      const enum BinaryOperation::flags & flags) noexcept
      : BinaryOperation({ srctype, srctype }, std::move(dsttype)),
        flags_(flags)
  {}

  virtual bool
  operator==(const Operation & other) const noexcept override;

  virtual rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  virtual rvsdg::Output *
  reduce_operand_pair(rvsdg::unop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum BinaryOperation::flags
  flags() const noexcept override;

  virtual std::string
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
    binary_op op(srctype, std::move(dsttype), BinaryOperation::flags::none);
    return &rvsdg::SimpleNode::Create(*op1->region(), op, { op1, op2 });
  }

  static inline rvsdg::Output *
  create_normalized(
      const std::shared_ptr<const rvsdg::Type> srctype,
      std::shared_ptr<const rvsdg::Type> dsttype,
      rvsdg::Output * op1,
      rvsdg::Output * op2)
  {
    return rvsdg::CreateOpNode<binary_op>({ op1, op2 }, srctype, std::move(dsttype), flags::none)
        .output(0);
  }

private:
  enum BinaryOperation::flags flags_;
};

/* structural operation */

class structural_op final : public rvsdg::StructuralOperation
{
public:
  virtual ~structural_op() noexcept;

  virtual std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;
};

class StructuralNodeArgument;
class StructuralNodeInput;
class StructuralNodeOutput;

class structural_node final : public rvsdg::StructuralNode
{
public:
  ~structural_node() override;

private:
  structural_node(rvsdg::Region * parent, size_t nsubregions)
      : rvsdg::StructuralNode(parent, nsubregions)
  {}

public:
  [[nodiscard]] const structural_op &
  GetOperation() const noexcept override;

  StructuralNodeInput &
  AddInput(rvsdg::Output & origin);

  StructuralNodeInput &
  AddInputWithArguments(rvsdg::Output & origin);

  StructuralNodeOutput &
  AddOutput(std::shared_ptr<const rvsdg::Type> type);

  StructuralNodeOutput &
  AddOutputWithResults(const std::vector<rvsdg::Output *> & origins);

  static structural_node *
  create(rvsdg::Region * parent, size_t nsubregions)
  {
    return new structural_node(parent, nsubregions);
  }

  virtual structural_node *
  copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const override;
};

class StructuralNodeInput final : public rvsdg::StructuralInput
{
  friend structural_node;

public:
  ~StructuralNodeInput() noexcept override;

private:
  StructuralNodeInput(
      structural_node & node,
      rvsdg::Output & origin,
      std::shared_ptr<const rvsdg::Type> type)
      : StructuralInput(&node, &origin, std::move(type))
  {}

public:
  [[nodiscard]] size_t
  NumArguments() const noexcept
  {
    return arguments.size();
  }

  [[nodiscard]] StructuralNodeArgument &
  Argument(size_t n) noexcept
  {
    JLM_ASSERT(n < NumArguments());
    // FIXME: I did not find a better way of doing it. The arguments attribute should be replaced
    // by a std::vector<> to enable efficient access.
    for (auto & argument : arguments)
    {
      if (argument.region()->index() == n)
        return *util::AssertedCast<StructuralNodeArgument>(&argument);
    }

    JLM_UNREACHABLE("Unknown argument");
  }
};

class StructuralNodeOutput final : public rvsdg::StructuralOutput
{
  friend structural_node;

public:
  ~StructuralNodeOutput() noexcept override;

private:
  StructuralNodeOutput(structural_node & node, std::shared_ptr<const rvsdg::Type> type)
      : StructuralOutput(&node, std::move(type))
  {}
};

class StructuralNodeArgument final : public rvsdg::RegionArgument
{
  friend structural_node;

public:
  ~StructuralNodeArgument() noexcept override;

  StructuralNodeArgument &
  Copy(rvsdg::Region & region, rvsdg::StructuralInput * input) override;

private:
  StructuralNodeArgument(
      rvsdg::Region & region,
      StructuralNodeInput * input,
      std::shared_ptr<const rvsdg::Type> type)
      : rvsdg::RegionArgument(&region, input, std::move(type))
  {}

  static StructuralNodeArgument &
  Create(rvsdg::Region & region, StructuralNodeInput & input)
  {
    auto argument = new StructuralNodeArgument(region, &input, input.Type());
    region.append_argument(argument);
    return *argument;
  }

  static StructuralNodeArgument &
  Create(rvsdg::Region & region, std::shared_ptr<const rvsdg::Type> type)
  {
    auto argument = new StructuralNodeArgument(region, nullptr, std::move(type));
    region.append_argument(argument);
    return *argument;
  }
};

class StructuralNodeResult final : public rvsdg::RegionResult
{
  friend structural_node;

public:
  ~StructuralNodeResult() noexcept override;

  StructuralNodeResult &
  Copy(rvsdg::Output & origin, rvsdg::StructuralOutput * output) override;

private:
  StructuralNodeResult(rvsdg::Output & origin, StructuralNodeOutput * output)
      : rvsdg::RegionResult(origin.region(), &origin, output, origin.Type())
  {}

  static StructuralNodeResult &
  Create(rvsdg::Output & origin)
  {
    auto result = new StructuralNodeResult(origin, nullptr);
    origin.region()->append_result(result);
    return *result;
  }

  static StructuralNodeResult &
  Create(rvsdg::Output & origin, StructuralNodeOutput & output)
  {
    auto result = new StructuralNodeResult(origin, &output);
    origin.region()->append_result(result);
    return *result;
  }
};

class test_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~test_op();

  inline test_op(
      std::vector<std::shared_ptr<const rvsdg::Type>> arguments,
      std::vector<std::shared_ptr<const rvsdg::Type>> results)
      : SimpleOperation(std::move(arguments), std::move(results))
  {}

  test_op(const test_op &) = default;

  virtual bool
  operator==(const Operation & other) const noexcept override;

  virtual std::string
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

    test_op op(std::move(operand_types), std::move(result_types));
    return &rvsdg::SimpleNode::Create(*region, op, { operands });
  }

  static rvsdg::SimpleNode *
  Create(
      rvsdg::Region * region,
      std::vector<std::shared_ptr<const rvsdg::Type>> operandTypes,
      const std::vector<rvsdg::Output *> & operands,
      std::vector<std::shared_ptr<const rvsdg::Type>> resultTypes)
  {
    test_op op(std::move(operandTypes), std::move(resultTypes));
    return &rvsdg::SimpleNode::Create(*region, op, { operands });
  }
};

static inline std::unique_ptr<llvm::ThreeAddressCode>
create_testop_tac(
    const std::vector<const llvm::variable *> & arguments,
    std::vector<std::shared_ptr<const rvsdg::Type>> result_types)
{
  std::vector<std::shared_ptr<const rvsdg::Type>> argument_types;
  for (const auto & arg : arguments)
    argument_types.push_back(arg->Type());

  test_op op(std::move(argument_types), std::move(result_types));
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

  return operands.empty() ? outputs(&rvsdg::CreateOpNode<test_op>(
                                *region,
                                std::move(operand_types),
                                std::move(result_types)))
                          : outputs(&rvsdg::CreateOpNode<test_op>(
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

class TestGraphResult final : public jlm::rvsdg::RegionResult
{
private:
  TestGraphResult(
      rvsdg::Region & region,
      jlm::rvsdg::Output & origin,
      rvsdg::StructuralOutput * output)
      : jlm::rvsdg::RegionResult(&region, &origin, output, origin.Type())
  {}

  TestGraphResult(rvsdg::Output & origin, rvsdg::StructuralOutput * output)
      : TestGraphResult(*origin.region(), origin, output)
  {}

public:
  TestGraphResult &
  Copy(rvsdg::Output & origin, rvsdg::StructuralOutput * output) override
  {
    return Create(origin, output);
  }

  static TestGraphResult &
  Create(rvsdg::Region & region, jlm::rvsdg::Output & origin, rvsdg::StructuralOutput * output)
  {
    auto graphResult = new TestGraphResult(region, origin, output);
    origin.region()->append_result(graphResult);
    return *graphResult;
  }

  static TestGraphResult &
  Create(rvsdg::Output & origin, rvsdg::StructuralOutput * output)
  {
    return Create(*origin.region(), origin, output);
  }
};

}

#endif

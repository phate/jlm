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
  GraphImport(rvsdg::graph & graph, std::shared_ptr<const rvsdg::Type> type, std::string name)
      : rvsdg::GraphImport(graph, std::move(type), std::move(name))
  {}

public:
  GraphImport &
  Copy(rvsdg::Region & region, rvsdg::structural_input * input) override;

  static GraphImport &
  Create(rvsdg::graph & graph, std::shared_ptr<const rvsdg::Type> type, std::string name)
  {
    auto graphImport = new GraphImport(graph, std::move(type), std::move(name));
    graph.root()->append_argument(graphImport);
    return *graphImport;
  }
};

/**
 * Represents an export from the RVSDG of an internal entity.
 * It can be used for testing of graph exports.
 */
class GraphExport final : public rvsdg::GraphExport
{
  GraphExport(rvsdg::output & origin, std::string name)
      : rvsdg::GraphExport(origin, std::move(name))
  {}

public:
  GraphExport &
  Copy(rvsdg::output & origin, rvsdg::structural_output * output) override;

  static GraphExport &
  Create(rvsdg::output & origin, std::string name)
  {
    auto graphExport = new GraphExport(origin, std::move(name));
    origin.region()->graph()->root()->append_result(graphExport);
    return *graphExport;
  }
};

/* unary operation */

class unary_op final : public rvsdg::unary_op
{
public:
  virtual ~unary_op() noexcept;

  inline unary_op(
      std::shared_ptr<const rvsdg::Type> srctype,
      std::shared_ptr<const rvsdg::Type> dsttype) noexcept
      : rvsdg::unary_op(std::move(srctype), std::move(dsttype))
  {}

  virtual bool
  operator==(const rvsdg::operation & other) const noexcept override;

  virtual rvsdg::unop_reduction_path_t
  can_reduce_operand(const rvsdg::output * operand) const noexcept override;

  virtual rvsdg::output *
  reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * operand) const override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<rvsdg::operation>
  copy() const override;

  static inline rvsdg::node *
  create(
      rvsdg::Region * region,
      std::shared_ptr<const rvsdg::Type> srctype,
      rvsdg::output * operand,
      std::shared_ptr<const rvsdg::Type> dsttype)
  {
    return rvsdg::simple_node::create(
        region,
        unary_op(std::move(srctype), std::move(dsttype)),
        { operand });
  }

  static inline rvsdg::output *
  create_normalized(
      std::shared_ptr<const rvsdg::Type> srctype,
      rvsdg::output * operand,
      std::shared_ptr<const rvsdg::Type> dsttype)
  {
    unary_op op(std::move(srctype), std::move(dsttype));
    return rvsdg::simple_node::create_normalized(operand->region(), op, { operand })[0];
  }
};

static inline bool
is_unary_op(const rvsdg::operation & op) noexcept
{
  return dynamic_cast<const unary_op *>(&op);
}

static inline bool
is_unary_node(const rvsdg::node * node) noexcept
{
  return jlm::rvsdg::is<unary_op>(node);
}

/* binary operation */

class binary_op final : public rvsdg::binary_op
{
public:
  virtual ~binary_op() noexcept;

  inline binary_op(
      const std::shared_ptr<const rvsdg::Type> & srctype,
      std::shared_ptr<const rvsdg::Type> dsttype,
      const enum rvsdg::binary_op::flags & flags) noexcept
      : rvsdg::binary_op({ srctype, srctype }, std::move(dsttype)),
        flags_(flags)
  {}

  virtual bool
  operator==(const rvsdg::operation & other) const noexcept override;

  virtual rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
      const noexcept override;

  virtual rvsdg::output *
  reduce_operand_pair(rvsdg::unop_reduction_path_t path, rvsdg::output * op1, rvsdg::output * op2)
      const override;

  virtual enum rvsdg::binary_op::flags
  flags() const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<rvsdg::operation>
  copy() const override;

  static inline rvsdg::node *
  create(
      const std::shared_ptr<const rvsdg::Type> & srctype,
      std::shared_ptr<const rvsdg::Type> dsttype,
      rvsdg::output * op1,
      rvsdg::output * op2)
  {
    binary_op op(srctype, std::move(dsttype), rvsdg::binary_op::flags::none);
    return rvsdg::simple_node::create(op1->region(), op, { op1, op2 });
  }

  static inline rvsdg::output *
  create_normalized(
      const std::shared_ptr<const rvsdg::Type> srctype,
      std::shared_ptr<const rvsdg::Type> dsttype,
      rvsdg::output * op1,
      rvsdg::output * op2)
  {
    binary_op op(srctype, std::move(dsttype), rvsdg::binary_op::flags::none);
    return rvsdg::simple_node::create_normalized(op1->region(), op, { op1, op2 })[0];
  }

private:
  enum rvsdg::binary_op::flags flags_;
};

/* structural operation */

class structural_op final : public rvsdg::structural_op
{
public:
  virtual ~structural_op() noexcept;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<rvsdg::operation>
  copy() const override;
};

class StructuralNodeArgument;
class StructuralNodeInput;
class StructuralNodeOutput;

class structural_node final : public rvsdg::structural_node
{
public:
  ~structural_node() override;

private:
  structural_node(rvsdg::Region * parent, size_t nsubregions)
      : rvsdg::structural_node(structural_op(), parent, nsubregions)
  {}

public:
  StructuralNodeInput &
  AddInput(rvsdg::output & origin);

  StructuralNodeInput &
  AddInputWithArguments(rvsdg::output & origin);

  StructuralNodeOutput &
  AddOutput(std::shared_ptr<const rvsdg::Type> type);

  StructuralNodeOutput &
  AddOutputWithResults(const std::vector<rvsdg::output *> & origins);

  static structural_node *
  create(rvsdg::Region * parent, size_t nsubregions)
  {
    return new structural_node(parent, nsubregions);
  }

  virtual structural_node *
  copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const override;
};

class StructuralNodeInput final : public rvsdg::structural_input
{
  friend structural_node;

public:
  ~StructuralNodeInput() noexcept override;

private:
  StructuralNodeInput(
      structural_node & node,
      rvsdg::output & origin,
      std::shared_ptr<const rvsdg::Type> type)
      : rvsdg::structural_input(&node, &origin, std::move(type))
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

class StructuralNodeOutput final : public rvsdg::structural_output
{
  friend structural_node;

public:
  ~StructuralNodeOutput() noexcept override;

private:
  StructuralNodeOutput(structural_node & node, std::shared_ptr<const rvsdg::Type> type)
      : rvsdg::structural_output(&node, std::move(type))
  {}
};

class StructuralNodeArgument final : public rvsdg::RegionArgument
{
  friend structural_node;

public:
  ~StructuralNodeArgument() noexcept override;

  StructuralNodeArgument &
  Copy(rvsdg::Region & region, rvsdg::structural_input * input) override;

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
  Copy(rvsdg::output & origin, rvsdg::structural_output * output) override;

private:
  StructuralNodeResult(rvsdg::output & origin, StructuralNodeOutput * output)
      : rvsdg::RegionResult(origin.region(), &origin, output, origin.Type())
  {}

  static StructuralNodeResult &
  Create(rvsdg::output & origin)
  {
    auto result = new StructuralNodeResult(origin, nullptr);
    origin.region()->append_result(result);
    return *result;
  }

  static StructuralNodeResult &
  Create(rvsdg::output & origin, StructuralNodeOutput & output)
  {
    auto result = new StructuralNodeResult(origin, &output);
    origin.region()->append_result(result);
    return *result;
  }
};

class test_op final : public rvsdg::simple_op
{
public:
  virtual ~test_op();

  inline test_op(
      std::vector<std::shared_ptr<const rvsdg::Type>> arguments,
      std::vector<std::shared_ptr<const rvsdg::Type>> results)
      : simple_op(std::move(arguments), std::move(results))
  {}

  test_op(const test_op &) = default;

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<rvsdg::operation>
  copy() const override;

  static rvsdg::simple_node *
  create(
      rvsdg::Region * region,
      const std::vector<rvsdg::output *> & operands,
      std::vector<std::shared_ptr<const rvsdg::Type>> result_types)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> operand_types;
    for (const auto & operand : operands)
      operand_types.push_back(operand->Type());

    test_op op(std::move(operand_types), std::move(result_types));
    return rvsdg::simple_node::create(region, op, { operands });
  }

  static rvsdg::simple_node *
  Create(
      rvsdg::Region * region,
      std::vector<std::shared_ptr<const rvsdg::Type>> operandTypes,
      const std::vector<rvsdg::output *> & operands,
      std::vector<std::shared_ptr<const rvsdg::Type>> resultTypes)
  {
    test_op op(std::move(operandTypes), std::move(resultTypes));
    return rvsdg::simple_node::create(region, op, { operands });
  }
};

class SimpleNode final : public rvsdg::simple_node
{
private:
  SimpleNode(
      rvsdg::Region & region,
      const test_op & operation,
      const std::vector<rvsdg::output *> & operands)
      : simple_node(&region, operation, operands)
  {}

public:
  using rvsdg::node::RemoveInputsWhere;

  using rvsdg::node::RemoveOutputsWhere;

  static SimpleNode &
  Create(
      rvsdg::Region & region,
      const std::vector<rvsdg::output *> & operands,
      std::vector<std::shared_ptr<const rvsdg::Type>> resultTypes)
  {
    auto operandTypes = ExtractTypes(operands);
    test_op operation(std::move(operandTypes), std::move(resultTypes));

    auto node = new SimpleNode(region, operation, operands);
    return *node;
  }

private:
  static std::vector<std::shared_ptr<const rvsdg::Type>>
  ExtractTypes(const std::vector<rvsdg::output *> & outputs)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types;
    types.reserve(outputs.size());
    for (auto output : outputs)
    {
      types.emplace_back(output->Type());
    }

    return types;
  }
};

static inline std::unique_ptr<llvm::tac>
create_testop_tac(
    const std::vector<const llvm::variable *> & arguments,
    std::vector<std::shared_ptr<const rvsdg::Type>> result_types)
{
  std::vector<std::shared_ptr<const rvsdg::Type>> argument_types;
  for (const auto & arg : arguments)
    argument_types.push_back(arg->Type());

  test_op op(std::move(argument_types), std::move(result_types));
  return llvm::tac::create(op, arguments);
}

static inline std::vector<rvsdg::output *>
create_testop(
    rvsdg::Region * region,
    const std::vector<rvsdg::output *> & operands,
    std::vector<std::shared_ptr<const rvsdg::Type>> result_types)
{
  std::vector<std::shared_ptr<const rvsdg::Type>> operand_types;
  for (const auto & operand : operands)
    operand_types.push_back(operand->Type());

  test_op op(std::move(operand_types), std::move(result_types));
  return rvsdg::simple_node::create_normalized(region, op, { operands });
}

class TestGraphArgument final : public jlm::rvsdg::RegionArgument
{
private:
  TestGraphArgument(
      rvsdg::Region & region,
      jlm::rvsdg::structural_input * input,
      std::shared_ptr<const jlm::rvsdg::Type> type)
      : jlm::rvsdg::RegionArgument(&region, input, type)
  {}

public:
  TestGraphArgument &
  Copy(rvsdg::Region & region, jlm::rvsdg::structural_input * input) override
  {
    return Create(region, input, Type());
  }

  static TestGraphArgument &
  Create(
      rvsdg::Region & region,
      jlm::rvsdg::structural_input * input,
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
      jlm::rvsdg::output & origin,
      jlm::rvsdg::structural_output * output)
      : jlm::rvsdg::RegionResult(&region, &origin, output, origin.Type())
  {}

  TestGraphResult(jlm::rvsdg::output & origin, jlm::rvsdg::structural_output * output)
      : TestGraphResult(*origin.region(), origin, output)
  {}

public:
  TestGraphResult &
  Copy(jlm::rvsdg::output & origin, jlm::rvsdg::structural_output * output) override
  {
    return Create(origin, output);
  }

  static TestGraphResult &
  Create(
      rvsdg::Region & region,
      jlm::rvsdg::output & origin,
      jlm::rvsdg::structural_output * output)
  {
    auto graphResult = new TestGraphResult(region, origin, output);
    origin.region()->append_result(graphResult);
    return *graphResult;
  }

  static TestGraphResult &
  Create(jlm::rvsdg::output & origin, jlm::rvsdg::structural_output * output)
  {
    return Create(*origin.region(), origin, output);
  }
};

}

#endif

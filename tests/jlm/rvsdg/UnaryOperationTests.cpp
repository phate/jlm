/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>

#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/unary.hpp>
#include <jlm/rvsdg/view.hpp>

class UnaryOperation final : public jlm::rvsdg::UnaryOperation
{
public:
  UnaryOperation(
      const std::shared_ptr<const jlm::rvsdg::Type> & operandType,
      const std::shared_ptr<const jlm::rvsdg::Type> & resultType)
      : jlm::rvsdg::UnaryOperation(operandType, resultType)
  {}

  jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::Output * operand) const noexcept override
  {

    if (const auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*operand);
        jlm::rvsdg::is<jlm::tests::NullaryOperation>(node))
    {
      return jlm::rvsdg::unop_reduction_constant;
    }

    return jlm::rvsdg::unop_reduction_none;
  }

  jlm::rvsdg::Output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::Output * operand)
      const override
  {
    if (path == jlm::rvsdg::unop_reduction_constant)
    {
      return operand;
    }

    return nullptr;
  }

  bool
  operator==(const Operation &) const noexcept override
  {
    JLM_UNREACHABLE("Not implemented.");
  }

  [[nodiscard]] std::string
  debug_string() const override
  {
    return "UnaryOperation";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<UnaryOperation>(this->argument(0), this->result(0));
  }
};

static void
NormalizeUnaryOperation_Success()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  const auto valueType = TestType::Create(TypeKind::Value);

  const auto nullaryNode =
      &CreateOpNode<jlm::tests::NullaryOperation>(graph.GetRootRegion(), valueType);

  const auto unaryNode =
      &CreateOpNode<::UnaryOperation>({ nullaryNode->output(0) }, valueType, valueType);

  auto & ex = GraphExport::Create(*unaryNode->output(0), "o2");

  view(graph, stdout);

  // Act
  const auto success = ReduceNode<::UnaryOperation>(NormalizeUnaryOperation, *unaryNode);
  view(graph, stdout);

  // Assert
  assert(success == true);

  graph.PruneNodes();
  assert(graph.GetRootRegion().numNodes() == 1);

  const auto node = TryGetOwnerNode<SimpleNode>(*ex.origin());
  assert(node == nullaryNode);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/UnaryOperationTests-NormalizeUnaryOperation_Success",
    NormalizeUnaryOperation_Success)

static void
NormalizeUnaryOperation_Failure()
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = TestType::Create(TypeKind::Value);

  Graph graph;
  auto i0 = &GraphImport::Create(graph, valueType, "i0");

  const auto unaryNode = &CreateOpNode<::UnaryOperation>({ i0 }, valueType, valueType);

  auto & ex = GraphExport::Create(*unaryNode->output(0), "o2");

  view(graph, stdout);

  // Act
  const auto success = ReduceNode<::UnaryOperation>(NormalizeUnaryOperation, *unaryNode);
  view(graph, stdout);

  // Assert
  assert(success == false);

  graph.PruneNodes();
  assert(graph.GetRootRegion().numNodes() == 1);

  const auto node = TryGetOwnerNode<SimpleNode>(*ex.origin());
  assert(node == unaryNode);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/UnaryOperationTests-NormalizeUnaryOperation_Failure",
    NormalizeUnaryOperation_Failure)

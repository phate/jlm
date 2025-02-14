/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/nullary.hpp>
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
  can_reduce_operand(const jlm::rvsdg::output * operand) const noexcept override
  {

    if (const auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*operand);
        jlm::rvsdg::is<jlm::tests::NullaryOperation>(node))
    {
      return jlm::rvsdg::unop_reduction_constant;
    }

    return jlm::rvsdg::unop_reduction_none;
  }

  jlm::rvsdg::output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::output * operand)
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

static int
NormalizeUnaryOperation_Success()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  const auto valueType = jlm::tests::valuetype::Create();

  const jlm::tests::NullaryOperation nullaryOperation(valueType);
  const auto nullaryNode = &SimpleNode::Create(graph.GetRootRegion(), nullaryOperation, {});

  const ::UnaryOperation unaryOperation(valueType, valueType);
  const auto unaryNode =
      &SimpleNode::Create(graph.GetRootRegion(), unaryOperation, { nullaryNode->output(0) });

  auto & ex = jlm::tests::GraphExport::Create(*unaryNode->output(0), "o2");

  view(graph, stdout);

  // Act
  const auto success = ReduceNode<::UnaryOperation>(NormalizeUnaryOperation, *unaryNode);
  view(graph, stdout);

  // Assert
  assert(success == true);

  graph.PruneNodes();
  assert(graph.GetRootRegion().nnodes() == 1);

  const auto node = TryGetOwnerNode<SimpleNode>(*ex.origin());
  assert(node == nullaryNode);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/UnaryOperationTests-NormalizeUnaryOperation_Success",
    NormalizeUnaryOperation_Success)

static int
NormalizeUnaryOperation_Failure()
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = jlm::tests::valuetype::Create();

  Graph graph;
  auto i0 = &jlm::tests::GraphImport::Create(graph, valueType, "i0");

  const ::UnaryOperation unaryOperation(valueType, valueType);
  const auto unaryNode = &SimpleNode::Create(graph.GetRootRegion(), unaryOperation, { i0 });

  auto & ex = jlm::tests::GraphExport::Create(*unaryNode->output(0), "o2");

  view(graph, stdout);

  // Act
  const auto success = ReduceNode<::UnaryOperation>(NormalizeUnaryOperation, *unaryNode);
  view(graph, stdout);

  // Assert
  assert(success == false);

  graph.PruneNodes();
  assert(graph.GetRootRegion().nnodes() == 1);

  const auto node = TryGetOwnerNode<SimpleNode>(*ex.origin());
  assert(node == unaryNode);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/UnaryOperationTests-NormalizeUnaryOperation_Failure",
    NormalizeUnaryOperation_Failure)

/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/view.hpp>

class BinaryOperation final : public jlm::rvsdg::BinaryOperation
{
public:
  BinaryOperation(
      const std::shared_ptr<const jlm::rvsdg::Type> operandType,
      const std::shared_ptr<const jlm::rvsdg::Type> resultType,
      const enum jlm::rvsdg::BinaryOperation::flags & flags)
      : jlm::rvsdg::BinaryOperation({ operandType, operandType }, resultType),
        Flags_(flags)
  {}

  jlm::rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const jlm::rvsdg::Output * operand1, const jlm::rvsdg::Output * operand2)
      const noexcept override
  {
    auto n1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*operand1);
    auto n2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*operand2);

    if (jlm::rvsdg::is<jlm::rvsdg::UnaryOperation>(n1)
        && jlm::rvsdg::is<jlm::rvsdg::UnaryOperation>(n2))
    {
      return 1;
    }

    return 0;
  }

  jlm::rvsdg::Output *
  reduce_operand_pair(
      jlm::rvsdg::unop_reduction_path_t path,
      jlm::rvsdg::Output *,
      jlm::rvsdg::Output * op2) const override
  {

    if (path == 1)
    {
      return op2;
    }

    return nullptr;
  }

  [[nodiscard]] enum jlm::rvsdg::BinaryOperation::flags
  flags() const noexcept override
  {
    return Flags_;
  }

  bool
  operator==(const Operation &) const noexcept override
  {
    JLM_UNREACHABLE("Not implemented.");
  }

  [[nodiscard]] std::string
  debug_string() const override
  {
    return "BinaryOperation";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<BinaryOperation>(this->argument(0), this->result(0), Flags_);
  }

private:
  enum jlm::rvsdg::BinaryOperation::flags Flags_;
};

static void
ReduceFlattenedBinaryReductionParallel()
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = jlm::tests::ValueType::Create();
  const jlm::tests::TestBinaryOperation binaryOperation(
      valueType,
      valueType,
      jlm::rvsdg::BinaryOperation::flags::associative);

  Graph graph;
  auto i0 = &jlm::tests::GraphImport::Create(graph, valueType, "");
  auto i1 = &jlm::tests::GraphImport::Create(graph, valueType, "");
  auto i2 = &jlm::tests::GraphImport::Create(graph, valueType, "");
  auto i3 = &jlm::tests::GraphImport::Create(graph, valueType, "");

  auto & node = CreateOpNode<FlattenedBinaryOperation>({ i0, i1, i2, i3 }, binaryOperation, 4);

  auto & ex = jlm::tests::GraphExport::Create(*node.output(0), "");

  view(graph, stdout);

  // Act
  FlattenedBinaryOperation::reduce(&graph, FlattenedBinaryOperation::reduction::parallel);
  graph.PruneNodes();
  view(graph, stdout);

  // Assert
  assert(graph.GetRootRegion().nnodes() == 3);

  auto node0 = TryGetOwnerNode<Node>(*ex.origin());
  assert(is<jlm::tests::TestBinaryOperation>(node0));

  auto node1 = TryGetOwnerNode<Node>(*node0->input(0)->origin());
  assert(is<jlm::tests::TestBinaryOperation>(node1));

  auto node2 = TryGetOwnerNode<Node>(*node0->input(1)->origin());
  assert(is<jlm::tests::TestBinaryOperation>(node2));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/test-binary-ReduceFlattenedBinaryReductionParallel",
    ReduceFlattenedBinaryReductionParallel)

static void
ReduceFlattenedBinaryReductionLinear()
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = jlm::tests::ValueType::Create();
  const jlm::tests::TestBinaryOperation binaryOperation(
      valueType,
      valueType,
      jlm::rvsdg::BinaryOperation::flags::associative);

  Graph graph;
  auto i0 = &jlm::tests::GraphImport::Create(graph, valueType, "");
  auto i1 = &jlm::tests::GraphImport::Create(graph, valueType, "");
  auto i2 = &jlm::tests::GraphImport::Create(graph, valueType, "");
  auto i3 = &jlm::tests::GraphImport::Create(graph, valueType, "");

  auto & node = CreateOpNode<FlattenedBinaryOperation>({ i0, i1, i2, i3 }, binaryOperation, 4);

  auto & ex = jlm::tests::GraphExport::Create(*node.output(0), "");

  view(graph, stdout);

  // Act
  FlattenedBinaryOperation::reduce(&graph, FlattenedBinaryOperation::reduction::linear);
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  assert(graph.GetRootRegion().nnodes() == 3);

  auto node0 = TryGetOwnerNode<Node>(*ex.origin());
  assert(is<jlm::tests::TestBinaryOperation>(node0));

  auto node1 = TryGetOwnerNode<Node>(*node0->input(0)->origin());
  assert(is<jlm::tests::TestBinaryOperation>(node1));

  auto node2 = TryGetOwnerNode<Node>(*node1->input(0)->origin());
  assert(is<jlm::tests::TestBinaryOperation>(node2));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/test-binary-ReduceFlattenedBinaryReductionLinear",
    ReduceFlattenedBinaryReductionLinear)

static void
FlattenAssociativeBinaryOperation_NotAssociativeBinary()
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();

  Graph graph;
  auto i0 = &jlm::tests::GraphImport::Create(graph, valueType, "i0");
  auto i1 = &jlm::tests::GraphImport::Create(graph, valueType, "i1");
  auto i2 = &jlm::tests::GraphImport::Create(graph, valueType, "i2");

  jlm::tests::TestBinaryOperation binaryOperation(
      valueType,
      valueType,
      jlm::rvsdg::BinaryOperation::flags::none);
  auto o1 = &SimpleNode::Create(graph.GetRootRegion(), binaryOperation, { i0, i1 });
  auto o2 = &SimpleNode::Create(graph.GetRootRegion(), binaryOperation, { o1->output(0), i2 });

  auto & ex = jlm::tests::GraphExport::Create(*o2->output(0), "o2");

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto node = TryGetOwnerNode<SimpleNode>(*ex.origin());
  auto success =
      ReduceNode<jlm::tests::TestBinaryOperation>(FlattenAssociativeBinaryOperation, *node);

  jlm::rvsdg::view(graph, stdout);

  // Assert
  assert(success == false);
  assert(TryGetOwnerNode<SimpleNode>(*ex.origin()) == node);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/test-binary-FlattenAssociatedBinaryOperation_NotAssociativeBinary",
    FlattenAssociativeBinaryOperation_NotAssociativeBinary)

static void
FlattenAssociativeBinaryOperation_NoNewOperands()
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();

  Graph graph;
  auto i0 = &jlm::tests::GraphImport::Create(graph, valueType, "i0");
  auto i1 = &jlm::tests::GraphImport::Create(graph, valueType, "i1");

  jlm::tests::TestUnaryOperation unaryOperation(valueType, valueType);
  jlm::tests::TestBinaryOperation binaryOperation(
      valueType,
      valueType,
      jlm::rvsdg::BinaryOperation::flags::associative);
  auto u1 = &SimpleNode::Create(graph.GetRootRegion(), unaryOperation, { i0 });
  auto u2 = &SimpleNode::Create(graph.GetRootRegion(), unaryOperation, { i1 });
  auto b2 =
      &SimpleNode::Create(graph.GetRootRegion(), binaryOperation, { u1->output(0), u2->output(0) });

  auto & ex = jlm::tests::GraphExport::Create(*b2->output(0), "o2");

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto node = TryGetOwnerNode<SimpleNode>(*ex.origin());
  auto success =
      ReduceNode<jlm::tests::TestBinaryOperation>(FlattenAssociativeBinaryOperation, *node);

  jlm::rvsdg::view(graph, stdout);

  // Assert
  assert(success == false);
  assert(TryGetOwnerNode<SimpleNode>(*ex.origin()) == node);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/test-binary-FlattenAssociatedBinaryOperation_NoNewOperands",
    FlattenAssociativeBinaryOperation_NoNewOperands)

static void
FlattenAssociativeBinaryOperation_Success()
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();

  Graph graph;
  auto i0 = &jlm::tests::GraphImport::Create(graph, valueType, "i0");
  auto i1 = &jlm::tests::GraphImport::Create(graph, valueType, "i1");
  auto i2 = &jlm::tests::GraphImport::Create(graph, valueType, "i2");

  jlm::tests::TestBinaryOperation binaryOperation(
      valueType,
      valueType,
      jlm::rvsdg::BinaryOperation::flags::associative);
  auto o1 = &SimpleNode::Create(graph.GetRootRegion(), binaryOperation, { i0, i1 });
  auto o2 = &SimpleNode::Create(graph.GetRootRegion(), binaryOperation, { o1->output(0), i2 });

  auto & ex = jlm::tests::GraphExport::Create(*o2->output(0), "o2");

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto node = TryGetOwnerNode<SimpleNode>(*ex.origin());
  auto success =
      ReduceNode<jlm::tests::TestBinaryOperation>(FlattenAssociativeBinaryOperation, *node);

  jlm::rvsdg::view(graph, stdout);

  // Assert
  assert(success);
  auto flattenedBinaryNode = TryGetOwnerNode<SimpleNode>(*ex.origin());
  assert(is<FlattenedBinaryOperation>(flattenedBinaryNode));
  assert(flattenedBinaryNode->ninputs() == 3);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/test-binary-FlattenAssociatedBinaryOperation_Success",
    FlattenAssociativeBinaryOperation_Success)

static void
NormalizeBinaryOperation_NoNewOperands()
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();

  Graph graph;
  auto i0 = &jlm::tests::GraphImport::Create(graph, valueType, "i0");
  auto i1 = &jlm::tests::GraphImport::Create(graph, valueType, "i1");

  jlm::tests::TestBinaryOperation binaryOperation(
      valueType,
      valueType,
      jlm::rvsdg::BinaryOperation::flags::associative);
  auto o1 = &SimpleNode::Create(graph.GetRootRegion(), binaryOperation, { i0, i1 });

  auto & ex = jlm::tests::GraphExport::Create(*o1->output(0), "o2");

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto node = TryGetOwnerNode<SimpleNode>(*ex.origin());
  auto success = ReduceNode<jlm::tests::TestBinaryOperation>(NormalizeBinaryOperation, *node);

  jlm::rvsdg::view(graph, stdout);

  // Assert
  assert(success == false);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/test-binary-NormalizeBinaryOperation_NoNewOperands",
    NormalizeBinaryOperation_NoNewOperands)

static void
NormalizeBinaryOperation_SingleOperand()
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();

  jlm::tests::TestUnaryOperation unaryOperation(valueType, valueType);
  ::BinaryOperation binaryOperation(valueType, valueType, jlm::rvsdg::BinaryOperation::flags::none);

  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, valueType, "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, valueType, "s1");

  auto u1 = &SimpleNode::Create(graph.GetRootRegion(), unaryOperation, { s0 });
  auto u2 = &SimpleNode::Create(graph.GetRootRegion(), unaryOperation, { s1 });

  auto o1 =
      &SimpleNode::Create(graph.GetRootRegion(), binaryOperation, { u1->output(0), u2->output(0) });

  auto & ex = jlm::tests::GraphExport::Create(*o1->output(0), "ex");

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto node = TryGetOwnerNode<SimpleNode>(*ex.origin());
  auto success = ReduceNode<::BinaryOperation>(NormalizeBinaryOperation, *node);

  jlm::rvsdg::view(graph, stdout);

  // Assert
  assert(success == true);
  assert(ex.origin() == u2->output(0));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/test-binary-NormalizeBinaryOperation_SingleOperand",
    NormalizeBinaryOperation_SingleOperand)

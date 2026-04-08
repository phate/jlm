/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
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

    if (n1 && n2 && jlm::rvsdg::is<jlm::rvsdg::UnaryOperation>(n1->GetOperation())
        && jlm::rvsdg::is<jlm::rvsdg::UnaryOperation>(n2->GetOperation()))
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

TEST(BinaryOperationTests, ReduceFlattenedBinaryReductionParallel)
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = TestType::createValueType();
  const TestBinaryOperation binaryOperation(
      valueType,
      valueType,
      jlm::rvsdg::BinaryOperation::flags::associative);

  Graph graph;
  auto i0 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "");
  auto i1 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "");
  auto i2 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "");
  auto i3 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "");

  auto & node = CreateOpNode<FlattenedBinaryOperation>({ i0, i1, i2, i3 }, binaryOperation, 4);

  auto & ex = GraphExport::Create(*node.output(0), "");

  view(graph, stdout);

  // Act
  FlattenedBinaryOperation::reduce(&graph, FlattenedBinaryOperation::reduction::parallel);
  graph.PruneNodes();
  view(graph, stdout);

  // Assert
  EXPECT_EQ(graph.GetRootRegion().numNodes(), 3u);

  auto node0 = TryGetOwnerNode<SimpleNode>(*ex.origin());
  EXPECT_TRUE(is<TestBinaryOperation>(node0->GetOperation()));

  auto node1 = TryGetOwnerNode<SimpleNode>(*node0->input(0)->origin());
  EXPECT_TRUE(is<TestBinaryOperation>(node1->GetOperation()));

  auto node2 = TryGetOwnerNode<SimpleNode>(*node0->input(1)->origin());
  EXPECT_TRUE(is<TestBinaryOperation>(node2->GetOperation()));
}

TEST(BinaryOperationTests, ReduceFlattenedBinaryReductionLinear)
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto valueType = TestType::createValueType();
  const TestBinaryOperation binaryOperation(
      valueType,
      valueType,
      jlm::rvsdg::BinaryOperation::flags::associative);

  Graph graph;
  auto i0 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "");
  auto i1 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "");
  auto i2 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "");
  auto i3 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "");

  auto & node = CreateOpNode<FlattenedBinaryOperation>({ i0, i1, i2, i3 }, binaryOperation, 4);

  auto & ex = GraphExport::Create(*node.output(0), "");

  view(graph, stdout);

  // Act
  FlattenedBinaryOperation::reduce(&graph, FlattenedBinaryOperation::reduction::linear);
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  EXPECT_EQ(graph.GetRootRegion().numNodes(), 3u);

  auto node0 = TryGetOwnerNode<SimpleNode>(*ex.origin());
  EXPECT_TRUE(is<TestBinaryOperation>(node0->GetOperation()));

  auto node1 = TryGetOwnerNode<SimpleNode>(*node0->input(0)->origin());
  EXPECT_TRUE(is<TestBinaryOperation>(node1->GetOperation()));

  auto node2 = TryGetOwnerNode<SimpleNode>(*node1->input(0)->origin());
  EXPECT_TRUE(is<TestBinaryOperation>(node2->GetOperation()));
}

TEST(BinaryOperationTests, FlattenAssociativeBinaryOperation_NotAssociativeBinary)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();

  Graph graph;
  auto i0 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "i0");
  auto i1 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "i1");
  auto i2 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "i2");

  auto o1 = &CreateOpNode<TestBinaryOperation>(
      { i0, i1 },
      valueType,
      valueType,
      jlm::rvsdg::BinaryOperation::flags::none);
  auto o2 = &CreateOpNode<TestBinaryOperation>(
      { o1->output(0), i2 },
      valueType,
      valueType,
      jlm::rvsdg::BinaryOperation::flags::none);

  auto & ex = GraphExport::Create(*o2->output(0), "o2");

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto node = TryGetOwnerNode<SimpleNode>(*ex.origin());
  auto success = ReduceNode<TestBinaryOperation>(FlattenAssociativeBinaryOperation, *node);

  jlm::rvsdg::view(graph, stdout);

  // Assert
  EXPECT_FALSE(success);
  EXPECT_EQ(TryGetOwnerNode<SimpleNode>(*ex.origin()), node);
}

TEST(BinaryOperationTests, FlattenAssociativeBinaryOperation_NoNewOperands)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();

  Graph graph;
  auto i0 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "i0");
  auto i1 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "i1");

  auto u1 = &CreateOpNode<TestUnaryOperation>({ i0 }, valueType, valueType);
  auto u2 = &CreateOpNode<TestUnaryOperation>({ i1 }, valueType, valueType);
  auto b2 = &CreateOpNode<TestBinaryOperation>(
      { u1->output(0), u2->output(0) },
      valueType,
      valueType,
      jlm::rvsdg::BinaryOperation::flags::associative);

  auto & ex = GraphExport::Create(*b2->output(0), "o2");

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto node = TryGetOwnerNode<SimpleNode>(*ex.origin());
  auto success = ReduceNode<TestBinaryOperation>(FlattenAssociativeBinaryOperation, *node);

  jlm::rvsdg::view(graph, stdout);

  // Assert
  EXPECT_FALSE(success);
  EXPECT_EQ(TryGetOwnerNode<SimpleNode>(*ex.origin()), node);
}

TEST(BinaryOperationTests, FlattenAssociativeBinaryOperation_Success)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();

  Graph graph;
  auto i0 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "i0");
  auto i1 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "i1");
  auto i2 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "i2");

  auto o1 = &CreateOpNode<TestBinaryOperation>(
      { i0, i1 },
      valueType,
      valueType,
      jlm::rvsdg::BinaryOperation::flags::associative);
  auto o2 = &CreateOpNode<TestBinaryOperation>(
      { o1->output(0), i2 },
      valueType,
      valueType,
      jlm::rvsdg::BinaryOperation::flags::associative);

  auto & ex = GraphExport::Create(*o2->output(0), "o2");

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto node = TryGetOwnerNode<SimpleNode>(*ex.origin());
  auto success = ReduceNode<TestBinaryOperation>(FlattenAssociativeBinaryOperation, *node);

  jlm::rvsdg::view(graph, stdout);

  // Assert
  EXPECT_TRUE(success);
  auto flattenedBinaryNode = TryGetOwnerNode<SimpleNode>(*ex.origin());
  EXPECT_TRUE(is<FlattenedBinaryOperation>(flattenedBinaryNode->GetOperation()));
  EXPECT_EQ(flattenedBinaryNode->ninputs(), 3u);
}

TEST(BinaryOperationTests, NormalizeBinaryOperation_NoNewOperands)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();

  Graph graph;
  auto i0 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "i0");
  auto i1 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "i1");

  auto o1 = &CreateOpNode<TestBinaryOperation>(
      { i0, i1 },
      valueType,
      valueType,
      jlm::rvsdg::BinaryOperation::flags::associative);

  auto & ex = GraphExport::Create(*o1->output(0), "o2");

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto node = TryGetOwnerNode<SimpleNode>(*ex.origin());
  auto success = ReduceNode<TestBinaryOperation>(NormalizeBinaryOperation, *node);

  jlm::rvsdg::view(graph, stdout);

  // Assert
  EXPECT_FALSE(success);
}

TEST(BinaryOperationTests, NormalizeBinaryOperation_SingleOperand)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();

  Graph graph;
  auto s0 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "s0");
  auto s1 = &jlm::rvsdg::GraphImport::Create(graph, valueType, "s1");

  auto u1 = &CreateOpNode<TestUnaryOperation>({ s0 }, valueType, valueType);
  auto u2 = &CreateOpNode<TestUnaryOperation>({ s1 }, valueType, valueType);

  auto o1 = &CreateOpNode<::BinaryOperation>(
      { u1->output(0), u2->output(0) },
      valueType,
      valueType,
      jlm::rvsdg::BinaryOperation::flags::none);

  auto & ex = GraphExport::Create(*o1->output(0), "ex");

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto node = TryGetOwnerNode<SimpleNode>(*ex.origin());
  auto success = ReduceNode<::BinaryOperation>(NormalizeBinaryOperation, *node);

  jlm::rvsdg::view(graph, stdout);

  // Assert
  EXPECT_TRUE(success);
  EXPECT_EQ(ex.origin(), u2->output(0));
}

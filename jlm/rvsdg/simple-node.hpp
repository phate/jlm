/*
 * Copyright 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_SIMPLE_NODE_HPP
#define JLM_RVSDG_SIMPLE_NODE_HPP

#include <jlm/rvsdg/node.hpp>

#include <optional>

namespace jlm::rvsdg
{

class SimpleOperation;

class SimpleNode final : public Node
{
public:
  ~SimpleNode() override;

private:
  SimpleNode(
      rvsdg::Region & region,
      std::unique_ptr<SimpleOperation> operation,
      const std::vector<jlm::rvsdg::Output *> & operands);

public:
  NodeInput *
  input(size_t index) const noexcept;

  NodeOutput *
  output(size_t index) const noexcept;

  [[nodiscard]] const SimpleOperation &
  GetOperation() const noexcept override;

  Node *
  copy(Region * region, const std::vector<Output *> & operands) const override;

  Node *
  copy(Region * region, SubstitutionMap & smap) const override;

  std::string
  DebugString() const override;

  static SimpleNode &
  Create(
      Region & region,
      std::unique_ptr<Operation> operation,
      const std::vector<rvsdg::Output *> & operands)
  {
    if (!is<SimpleOperation>(*operation))
      throw util::Error("Expected operation derived from SimpleOperation");

    std::unique_ptr<SimpleOperation> simpleOperation(
        util::assertedCast<SimpleOperation>(operation.release()));
    return *new SimpleNode(region, std::move(simpleOperation), operands);
  }

private:
  std::unique_ptr<SimpleOperation> Operation_;
};

/**
 * \brief Performs common node elimination for a given operation and operands in a region.
 *
 * @param region The region in which common node elimination is performed.
 * @param operation The simple operation on which the transformation is performed.
 * @param operands The operands of the simple node.
 * @return If the normalization could be applied, then the results of the binary operation after
 * the transformation. Otherwise, std::nullopt.
 */
std::optional<std::vector<rvsdg::Output *>>
NormalizeSimpleOperationCommonNodeElimination(
    Region & region,
    const SimpleOperation & operation,
    const std::vector<rvsdg::Output *> & operands);

inline NodeInput *
SimpleNode::input(size_t index) const noexcept
{
  return Node::input(index);
}

inline NodeOutput *
SimpleNode::output(size_t index) const noexcept
{
  return Node::output(index);
}

/**
 * \brief Creates a simple node characterized by its operator.
 *
 * \tparam OperatorType
 *   The type of operator wrapped by the node.
 *
 * \tparam OperatorArguments
 *   Argument types of the operator to be constructed (should be
 *   implied, just specify the OperatorType).
 *
 * \param operands
 *   The operands to the operator (i.e. inputs to the node to be constructed).
 *
 * \param operatorArguments
 *   Constructor arguments for the operator to be constructed.
 *
 * \returns
 *   Reference to the node constructed.
 *
 * \pre
 *   \p operands must be non-empty, must be in the same region, and their
 *   types must match the operator constructed by this call.
 *
 * Constructs a new operator of type \p OperatorType using \p operatorArguments
 * as constructor arguments. Creates a simple node using the constructed operator
 * and the given \p operands as operands to the constructed operator.
 *
 * Usage example:
 * \code
 *   auto element_ptr = CreateOpNode<GetElementPtrOperation>(
 *     { ptr }, offsetTypes, pointeeTypes).outputs(0);
 * \endcode
 */
template<typename OperatorType, typename... OperatorArguments>
SimpleNode &
CreateOpNode(const std::vector<Output *> & operands, OperatorArguments... operatorArguments)
{
  JLM_ASSERT(!operands.empty());
  return SimpleNode::Create(
      *operands[0]->region(),
      std::make_unique<OperatorType>(std::move(operatorArguments)...),
      operands);
}

/**
 * \brief Creates a simple node characterized by its operator.
 *
 * \tparam OperatorType
 *   The type of operator wrapped by the node.
 *
 * \tparam OperatorArguments
 *   Argument types of the operator to be constructed (should be
 *   implied, just specify the OperatorType).
 *
 * \param region
 *   The region to create the node in.
 *
 * \param operatorArguments
 *   Constructor arguments for the operator to be constructed.
 *
 * \returns
 *   Reference to the node constructed.
 *
 * \pre
 *   The given operator must not take any operands.
 *
 * Constructs a new operator of type \p OperatorType using \p operatorArguments
 * as constructor arguments. Creates a simple node using the constructed operator
 * with no operands in the specified region.
 *
 * Usage example:
 * \code
 *   auto val = CreateOpNode<IntegerConstantOperation>(region, 42).outputs(0);
 * \endcode
 */
template<typename OperatorType, typename... OperatorArguments>
SimpleNode &
CreateOpNode(Region & region, OperatorArguments... operatorArguments)
{
  return SimpleNode::Create(
      region,
      std::make_unique<OperatorType>(std::move(operatorArguments)...),
      {});
}

/**
 * \brief Checks if this is an input to a \ref SimpleNode and of specified operation type.
 *
 * \tparam TOperation
 *   The operation type to be matched against.
 *
 * \param input
 *   Input to be checked.
 *
 * \returns
 *   A pair of the owning simple node and requested operation. If the owner of \p input is not a
 *   \ref SimpleNode, then <nullptr, nullptr> is returned. If the owner of \p input is a \ref
 * SimpleNode but not of the correct operation type, then <SimpleNode*, nullptr> are returned.
 * Otherwise, <SimpleNode*, TOperation*> are returned.
 *
 * Checks if the specified input belongs to a \ref SimpleNode of requested operation type.
 * If this is the case, returns a pair of pointers to the node and operation of matched type.
 *
 * See \ref def_use_inspection.
 */
template<typename TOperation>
[[nodiscard]] std::pair<SimpleNode *, const TOperation *>
TryGetSimpleNodeAndOptionalOp(const Input & input) noexcept
{
  const auto simpleNode = TryGetOwnerNode<SimpleNode>(input);
  if (!simpleNode)
  {
    return std::make_pair(nullptr, nullptr);
  }

  if (auto operation = dynamic_cast<const TOperation *>(&simpleNode->GetOperation()))
  {
    return std::make_pair(simpleNode, operation);
  }

  return std::make_pair(simpleNode, nullptr);
}

/**
 * \brief Checks if this is an output to a \ref SimpleNode and of specified operation type.
 *
 * \tparam TOperation
 *   The operation type to be matched against.
 *
 * \param output
 *   Output to be checked.
 *
 * \returns
 *   A pair of the owning simple node and requested operation. If the owner of \p output is not a
 *   \ref SimpleNode, then <nullptr, nullptr> is returned. If the owner of \p output is a \ref
 *   SimpleNode but not of the correct operation type, then <SimpleNode*, nullptr> are returned.
 *   Otherwise, <SimpleNode*, TOperation*> are returned.
 *
 * Checks if the specified output belongs to a \ref SimpleNode of requested operation type.
 * If this is the case, returns a pair of pointers to the node and operation of matched type.
 *
 * See \ref def_use_inspection.
 */
template<typename TOperation>
[[nodiscard]] std::pair<SimpleNode *, const TOperation *>
TryGetSimpleNodeAndOptionalOp(const Output & output) noexcept
{
  const auto simpleNode = TryGetOwnerNode<SimpleNode>(output);
  if (!simpleNode)
  {
    return std::make_pair(nullptr, nullptr);
  }

  if (auto operation = dynamic_cast<const TOperation *>(&simpleNode->GetOperation()))
  {
    return std::make_pair(simpleNode, operation);
  }

  return std::make_pair(simpleNode, nullptr);
}

/**
 * \brief Checks if the node is a \ref SimpleNode of the specified operation type.
 *
 * \tparam TOperation
 *   The operation type to be matched against.
 *
 * \param node
 *   Node to be checked.
 *
 * \returns
 *   A pair of the simple node and requested operation. If \p node is not a \ref SimpleNode,
 *   then <nullptr, nullptr> is returned. If \p is a \ref SimpleNode of a different operation type,
 *   then <SimpleNode*, nullptr> is returned. Otherwise, <SimpleNode*, TOperation*> is returned.
 *
 * Checks if the specified \p node is a \ref SimpleNode of the requested operation type.
 * If this is the case, returns a pair of pointers to the SimpleNode and operation.
 *
 * See \ref def_use_inspection.
 */
template<typename TOperation>
[[nodiscard]] std::pair<const SimpleNode *, const TOperation *>
TryGetSimpleNodeAndOptionalOp(const Node & node) noexcept
{
  const auto simpleNode = dynamic_cast<const SimpleNode *>(&node);
  if (!simpleNode)
  {
    return std::make_pair(nullptr, nullptr);
  }
  if (auto operation = dynamic_cast<const TOperation *>(&simpleNode->GetOperation()))
  {
    return std::make_pair(simpleNode, operation);
  }
  return std::make_pair(simpleNode, nullptr);
}

}

#endif

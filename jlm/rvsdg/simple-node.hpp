/*
 * Copyright 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_SIMPLE_NODE_HPP
#define JLM_RVSDG_SIMPLE_NODE_HPP

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/simple-normal-form.hpp>

namespace jlm::rvsdg
{

class SimpleOperation;
class simple_input;
class simple_output;

class SimpleNode : public Node
{
public:
  ~SimpleNode() override;

protected:
  SimpleNode(
      rvsdg::Region * region,
      const SimpleOperation & op,
      const std::vector<jlm::rvsdg::output *> & operands);

  SimpleNode(
      rvsdg::Region & region,
      std::unique_ptr<SimpleOperation> operation,
      const std::vector<jlm::rvsdg::output *> & operands);

public:
  jlm::rvsdg::simple_input *
  input(size_t index) const noexcept;

  jlm::rvsdg::simple_output *
  output(size_t index) const noexcept;

  [[nodiscard]] const SimpleOperation &
  GetOperation() const noexcept override;

  Node *
  copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::output *> & operands) const override;

  Node *
  copy(rvsdg::Region * region, SubstitutionMap & smap) const override;

  static inline jlm::rvsdg::SimpleNode *
  create(
      rvsdg::Region * region,
      const SimpleOperation & op,
      const std::vector<jlm::rvsdg::output *> & operands)
  {
    return new SimpleNode(region, op, operands);
  }

  static inline jlm::rvsdg::SimpleNode &
  Create(
      rvsdg::Region & region,
      std::unique_ptr<SimpleOperation> operation,
      const std::vector<jlm::rvsdg::output *> & operands)
  {
    return *new SimpleNode(region, std::move(operation), operands);
  }

  static inline std::vector<jlm::rvsdg::output *>
  create_normalized(
      rvsdg::Region * region,
      const SimpleOperation & op,
      const std::vector<jlm::rvsdg::output *> & operands)
  {
    auto nf = static_cast<simple_normal_form *>(region->graph()->GetNodeNormalForm(typeid(op)));
    return nf->normalized_create(region, op, operands);
  }
};

/* inputs */

class simple_input final : public node_input
{
  friend class jlm::rvsdg::output;

public:
  virtual ~simple_input() noexcept;

  simple_input(
      SimpleNode * node,
      jlm::rvsdg::output * origin,
      std::shared_ptr<const rvsdg::Type> type);

public:
  SimpleNode *
  node() const noexcept
  {
    return static_cast<SimpleNode *>(node_input::node());
  }
};

/* outputs */

class simple_output final : public node_output
{
  friend class jlm::rvsdg::simple_input;

public:
  virtual ~simple_output() noexcept;

  simple_output(jlm::rvsdg::SimpleNode * node, std::shared_ptr<const rvsdg::Type> type);

public:
  SimpleNode *
  node() const noexcept
  {
    return static_cast<SimpleNode *>(node_output::node());
  }
};

/* simple node method definitions */

inline jlm::rvsdg::simple_input *
SimpleNode::input(size_t index) const noexcept
{
  return static_cast<simple_input *>(Node::input(index));
}

inline jlm::rvsdg::simple_output *
SimpleNode::output(size_t index) const noexcept
{
  return static_cast<simple_output *>(Node::output(index));
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
CreateOpNode(const std::vector<output *> & operands, OperatorArguments... operatorArguments)
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

}

#endif

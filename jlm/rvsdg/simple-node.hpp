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
      rvsdg::Region * region,
      std::unique_ptr<SimpleOperation> op,
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

  static inline jlm::rvsdg::SimpleNode *
  Create(
      rvsdg::Region * region,
      std::unique_ptr<SimpleOperation> op,
      const std::vector<jlm::rvsdg::output *> & operands)
  {
    return new SimpleNode(region, std::move(op), operands);
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

struct CreateOpNodeResult
{
  Node * node;
  std::vector<output *> outputs;
};

template<typename OpType, typename... OpParameters>
CreateOpNodeResult
CreateOpNode(const std::vector<output *> & arguments, OpParameters... opParameters)
{
  Region * region = arguments.at(0)->region();
  for (const auto & argument : arguments)
  {
    JLM_ASSERT(argument->region() == region);
  }
  Node * node =
      SimpleNode::Create(region, std::make_unique<OpType>(std::move(opParameters)...), arguments);
  std::vector<output *> outputs;
  for (std::size_t n = 0; n < node->noutputs(); ++n)
  {
    outputs.push_back(node->output(n));
  }
  return { node, std::move(outputs) };
}

}

#endif

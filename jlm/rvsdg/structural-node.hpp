/*
 * Copyright 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_STRUCTURAL_NODE_HPP
#define JLM_RVSDG_STRUCTURAL_NODE_HPP

#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jlm::rvsdg
{

/* structural node */

class structural_input;
class structural_op;
class structural_output;

class StructuralNode : public node
{
public:
  ~StructuralNode() noexcept override;

protected:
  StructuralNode(
      /* FIXME: use move semantics instead of copy semantics for op */
      const jlm::rvsdg::structural_op & op,
      rvsdg::Region * region,
      size_t nsubregions);

public:
  inline size_t
  nsubregions() const noexcept
  {
    return subregions_.size();
  }

  [[nodiscard]] rvsdg::Region *
  subregion(size_t index) const noexcept
  {
    JLM_ASSERT(index < nsubregions());
    return subregions_[index].get();
  }

  inline jlm::rvsdg::structural_input *
  input(size_t index) const noexcept;

  inline jlm::rvsdg::structural_output *
  output(size_t index) const noexcept;

  structural_input *
  append_input(std::unique_ptr<structural_input> input);

  structural_output *
  append_output(std::unique_ptr<structural_output> output);

  using node::RemoveInput;

  using node::RemoveOutput;

private:
  std::vector<std::unique_ptr<rvsdg::Region>> subregions_;
};

/* structural input class */

typedef jlm::util::intrusive_list<RegionArgument, RegionArgument::structural_input_accessor>
    argument_list;

class structural_input : public node_input
{
  friend StructuralNode;

public:
  virtual ~structural_input() noexcept;

  structural_input(
      StructuralNode * node,
      jlm::rvsdg::output * origin,
      std::shared_ptr<const rvsdg::Type> type);

  static structural_input *
  create(
      StructuralNode * node,
      jlm::rvsdg::output * origin,
      std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    auto input = std::make_unique<structural_input>(node, origin, std::move(type));
    return node->append_input(std::move(input));
  }

  StructuralNode *
  node() const noexcept
  {
    return static_cast<StructuralNode *>(node_input::node());
  }

  argument_list arguments;
};

/* structural output class */

typedef jlm::util::intrusive_list<RegionResult, RegionResult::structural_output_accessor>
    result_list;

class structural_output : public node_output
{
  friend StructuralNode;

public:
  virtual ~structural_output() noexcept;

  structural_output(StructuralNode * node, std::shared_ptr<const rvsdg::Type> type);

  static structural_output *
  create(StructuralNode * node, std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    auto output = std::make_unique<structural_output>(node, std::move(type));
    return node->append_output(std::move(output));
  }

  StructuralNode *
  node() const noexcept
  {
    return static_cast<StructuralNode *>(node_output::node());
  }

  result_list results;
};

/* structural node method definitions */

inline jlm::rvsdg::structural_input *
StructuralNode::input(size_t index) const noexcept
{
  return static_cast<structural_input *>(node::input(index));
}

inline jlm::rvsdg::structural_output *
StructuralNode::output(size_t index) const noexcept
{
  return static_cast<structural_output *>(node::output(index));
}

template<class Operation>
bool
Region::Contains(const rvsdg::Region & region, bool checkSubregions)
{
  for (auto & node : region.nodes)
  {
    if (is<Operation>(&node))
    {
      return true;
    }

    if (!checkSubregions)
    {
      continue;
    }

    if (auto structuralNode = dynamic_cast<const StructuralNode *>(&node))
    {
      for (size_t n = 0; n < structuralNode->nsubregions(); n++)
      {
        if (Contains<Operation>(*structuralNode->subregion(n), checkSubregions))
        {
          return true;
        }
      }
    }
  }

  return false;
}

}

#endif

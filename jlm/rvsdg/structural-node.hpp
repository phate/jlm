/*
 * Copyright 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_STRUCTURAL_NODE_HPP
#define JLM_RVSDG_STRUCTURAL_NODE_HPP

#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/util/IteratorWrapper.hpp>

namespace jlm::rvsdg
{

/* structural node */

class StructuralInput;
class StructuralOperation;
class StructuralOutput;

class StructuralNode : public Node
{
  using SubregionIterator =
      util::PtrIterator<Region, std::vector<std::unique_ptr<Region>>::iterator>;
  using SubregionConstIterator =
      util::PtrIterator<const Region, std::vector<std::unique_ptr<Region>>::const_iterator>;

  using SubregionIteratorRange = util::IteratorRange<SubregionIterator>;
  using SubregionConstIteratorRange = util::IteratorRange<SubregionConstIterator>;

public:
  ~StructuralNode() noexcept override;

protected:
  StructuralNode(rvsdg::Region * region, size_t nsubregions);

public:
  std::string
  DebugString() const override;

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

  SubregionIteratorRange
  Subregions()
  {
    return { SubregionIterator(subregions_.begin()), SubregionIterator(subregions_.end()) };
  }

  SubregionConstIteratorRange
  Subregions() const
  {
    return { SubregionConstIterator(subregions_.begin()),
             SubregionConstIterator(subregions_.end()) };
  }

  [[nodiscard]] inline StructuralInput *
  input(size_t index) const noexcept;

  [[nodiscard]] inline StructuralOutput *
  output(size_t index) const noexcept;

  StructuralInput *
  append_input(std::unique_ptr<StructuralInput> input);

  StructuralOutput *
  append_output(std::unique_ptr<StructuralOutput> output);

  using Node::RemoveInput;

  using Node::RemoveOutput;

private:
  std::vector<std::unique_ptr<rvsdg::Region>> subregions_;
};

/* structural input class */

typedef jlm::util::IntrusiveList<RegionArgument, RegionArgument::structural_input_accessor>
    argument_list;

class StructuralInput : public NodeInput
{
  friend StructuralNode;

public:
  ~StructuralInput() noexcept override;

  StructuralInput(
      StructuralNode * node,
      jlm::rvsdg::Output * origin,
      std::shared_ptr<const rvsdg::Type> type);

  static StructuralInput *
  create(
      StructuralNode * node,
      jlm::rvsdg::Output * origin,
      std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    auto input = std::make_unique<StructuralInput>(node, origin, std::move(type));
    return node->append_input(std::move(input));
  }

  StructuralNode *
  node() const noexcept
  {
    return static_cast<StructuralNode *>(NodeInput::node());
  }

  argument_list arguments;
};

/* structural output class */

typedef jlm::util::IntrusiveList<RegionResult, RegionResult::structural_output_accessor>
    result_list;

class StructuralOutput : public node_output
{
  friend StructuralNode;

public:
  ~StructuralOutput() noexcept override;

  StructuralOutput(StructuralNode * node, std::shared_ptr<const rvsdg::Type> type);

  static StructuralOutput *
  create(StructuralNode * node, std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    auto output = std::make_unique<StructuralOutput>(node, std::move(type));
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

inline StructuralInput *
StructuralNode::input(size_t index) const noexcept
{
  return static_cast<StructuralInput *>(Node::input(index));
}

inline StructuralOutput *
StructuralNode::output(size_t index) const noexcept
{
  return static_cast<StructuralOutput *>(Node::output(index));
}

template<class Operation>
bool
Region::ContainsOperation(const rvsdg::Region & region, bool checkSubregions)
{
  for (auto & node : region.Nodes())
  {
    if (auto simpleNode = dynamic_cast<const SimpleNode *>(&node))
    {
      if (is<Operation>(simpleNode->GetOperation()))
      {
        return true;
      }
    }

    if (!checkSubregions)
    {
      continue;
    }

    if (auto structuralNode = dynamic_cast<const StructuralNode *>(&node))
    {
      for (size_t n = 0; n < structuralNode->nsubregions(); n++)
      {
        if (ContainsOperation<Operation>(*structuralNode->subregion(n), checkSubregions))
        {
          return true;
        }
      }
    }
  }

  return false;
}

template<class NodeType>
bool
Region::ContainsNodeType(const rvsdg::Region & region, bool checkSubregions)
{
  for (auto & node : region.Nodes())
  {
    if (dynamic_cast<const NodeType *>(&node))
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
        if (ContainsNodeType<NodeType>(*structuralNode->subregion(n), checkSubregions))
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

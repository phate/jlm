/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_DOMTREE_HPP
#define JLM_LLVM_IR_DOMTREE_HPP

#include <jlm/util/common.hpp>

#include <memory>
#include <vector>

namespace jlm::llvm
{

class ControlFlowGraph;
class ControlFlowGraphNode;

class DominatorTreeNode final
{
  typedef std::vector<std::unique_ptr<DominatorTreeNode>>::const_iterator const_iterator;

public:
  explicit DominatorTreeNode(ControlFlowGraphNode * node)
      : depth_(0),
        node_(node),
        parent_(nullptr)
  {}

  DominatorTreeNode(const DominatorTreeNode &) = delete;

  DominatorTreeNode(DominatorTreeNode &&) = delete;

  DominatorTreeNode &
  operator=(const DominatorTreeNode &) = delete;

  DominatorTreeNode &
  operator=(DominatorTreeNode &&) = delete;

  const_iterator
  begin() const
  {
    return children_.begin();
  }

  const_iterator
  end() const
  {
    return children_.end();
  }

  DominatorTreeNode *
  add_child(std::unique_ptr<DominatorTreeNode> child);

  size_t
  nchildren() const noexcept
  {
    return children_.size();
  }

  [[nodiscard]] DominatorTreeNode *
  child(size_t index) const noexcept
  {
    JLM_ASSERT(index < nchildren());
    return children_[index].get();
  }

  ControlFlowGraphNode *
  node() const noexcept
  {
    return node_;
  }

  [[nodiscard]] DominatorTreeNode *
  parent() const noexcept
  {
    return parent_;
  }

  size_t
  depth() const noexcept
  {
    return depth_;
  }

  static std::unique_ptr<DominatorTreeNode>
  create(ControlFlowGraphNode * node)
  {
    return std::unique_ptr<DominatorTreeNode>(new DominatorTreeNode(node));
  }

private:
  size_t depth_;
  ControlFlowGraphNode * node_;
  DominatorTreeNode * parent_;
  std::vector<std::unique_ptr<DominatorTreeNode>> children_;
};

std::unique_ptr<DominatorTreeNode>
domtree(ControlFlowGraph & cfg);

}

#endif

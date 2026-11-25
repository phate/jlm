/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_MODREFSETFOLDING_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_MODREFSETFOLDING_HPP

#include <jlm/util/iterator_range.hpp>
#include <jlm/util/IteratorWrapper.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace jlm::rvsdg
{
class Node;
class Region;
class RvsdgModule;
}

namespace jlm::llvm::aa
{

class ModRefSet;
class ModRefSummary;

class MemoryStateGraph
{
public:
  struct Node
  {
    Node(const rvsdg::Node & rvsdgNode, const ModRefSet & modRefSet)
        : rvsdgNode(rvsdgNode),
          modRefSet(modRefSet)
    {}

    const rvsdg::Node & rvsdgNode;
    const ModRefSet & modRefSet;
  };

  using NodeIterator =
      util::PtrIterator<const Node, std::vector<std::unique_ptr<Node>>::const_iterator>;
  using NodeRange = util::IteratorRange<NodeIterator>;
  using EdgeRange =
      util::IteratorRange<std::unordered_map<const Node *, const Node *>::const_iterator>;

  MemoryStateGraph(const rvsdg::Region & region, const ModRefSet & modRefSet);

  [[nodiscard]] const Node &
  getEntryNode() const noexcept;

  [[nodiscard]] const Node &
  getExitNode() const noexcept;

  [[nodiscard]] const rvsdg::Region &
  getRegion() const noexcept;

  [[nodiscard]] NodeRange
  nodes() const noexcept;

  [[nodiscard]] EdgeRange
  edges() const noexcept;

  Node &
  addNode(const rvsdg::Node & rvsdgNode, const ModRefSet & modRefSet);
  void
  addEdge(const Node & from, const Node & to);

  static std::unique_ptr<MemoryStateGraph>
  create(const rvsdg::Region & region, const ModRefSet & modRefSet);

private:
  std::vector<std::unique_ptr<Node>> nodes_;
  std::unordered_map<const Node *, const Node *> edges_;
  Node * entryNode_ = nullptr;
  Node * exitNode_ = nullptr;
  const rvsdg::Region * region_;
};

std::unique_ptr<MemoryStateGraph>
constructMemoryStateGraph(const rvsdg::Region & region, const ModRefSummary & modRefSummary);

std::string
toDot(const MemoryStateGraph & memoryStateGraph);

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_MODREFSETFOLDING_HPP

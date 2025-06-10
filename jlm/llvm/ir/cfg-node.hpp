/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_CFG_NODE_HPP
#define JLM_LLVM_IR_CFG_NODE_HPP

#include <jlm/util/common.hpp>
#include <jlm/util/iterator_range.hpp>
#include <jlm/util/IteratorWrapper.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

namespace jlm::llvm
{

class BasicBlock;
class ControlFlowGraph;
class cfg_node;

class ControlFlowGraphEdge final
{
public:
  ~ControlFlowGraphEdge() noexcept = default;

  ControlFlowGraphEdge(cfg_node * source, cfg_node * sink, size_t index) noexcept;

  ControlFlowGraphEdge(const ControlFlowGraphEdge & other) = delete;

  ControlFlowGraphEdge(ControlFlowGraphEdge && other) = default;

  ControlFlowGraphEdge &
  operator=(const ControlFlowGraphEdge & other) = delete;

  ControlFlowGraphEdge &
  operator=(ControlFlowGraphEdge && other) = default;

  void
  divert(cfg_node * new_sink);

  BasicBlock *
  split();

  cfg_node *
  source() const noexcept
  {
    return source_;
  }

  cfg_node *
  sink() const noexcept
  {
    return sink_;
  }

  /**
   * @return the index of this edge among the source's out edges
   */
  size_t
  index() const noexcept
  {
    return index_;
  }

  bool
  is_selfloop() const noexcept
  {
    return source_ == sink_;
  }

private:
  cfg_node * source_;
  cfg_node * sink_;
  size_t index_;

  friend cfg_node;
};

class cfg_node
{
  using inedge_iterator = util::
      PtrIterator<ControlFlowGraphEdge, std::unordered_set<ControlFlowGraphEdge *>::const_iterator>;
  using inedge_iterator_range = util::IteratorRange<inedge_iterator>;

  using outedge_iterator = util::PtrIterator<
      ControlFlowGraphEdge,
      std::vector<std::unique_ptr<ControlFlowGraphEdge>>::const_iterator>;
  using outedge_iterator_range = util::IteratorRange<outedge_iterator>;

public:
  virtual ~cfg_node();

protected:
  explicit cfg_node(ControlFlowGraph & cfg)
      : cfg_(cfg)
  {}

public:
  ControlFlowGraph &
  cfg() const noexcept
  {
    return cfg_;
  }

  size_t
  NumOutEdges() const noexcept;

  [[nodiscard]] ControlFlowGraphEdge *
  OutEdge(size_t n) const
  {
    JLM_ASSERT(n < NumOutEdges());
    return outedges_[n].get();
  }

  outedge_iterator_range
  OutEdges() const
  {
    return outedge_iterator_range(
        outedge_iterator(outedges_.begin()),
        outedge_iterator(outedges_.end()));
  }

  ControlFlowGraphEdge *
  add_outedge(cfg_node * sink)
  {
    outedges_.push_back(std::make_unique<ControlFlowGraphEdge>(this, sink, NumOutEdges()));
    sink->inedges_.insert(outedges_.back().get());
    return outedges_.back().get();
  }

  void
  remove_outedge(size_t n)
  {
    JLM_ASSERT(n < NumOutEdges());
    auto edge = outedges_[n].get();

    edge->sink()->inedges_.erase(edge);
    for (size_t i = n + 1; i < NumOutEdges(); i++)
    {
      outedges_[i - 1] = std::move(outedges_[i]);
      outedges_[i - 1]->index_ = outedges_[i - 1]->index_ - 1;
    }
    outedges_.resize(NumOutEdges() - 1);
  }

  void
  remove_outedges()
  {
    while (NumOutEdges() != 0)
      remove_outedge(NumOutEdges() - 1);
  }

  size_t
  NumInEdges() const noexcept;

  inedge_iterator_range
  InEdges() const
  {
    return inedge_iterator_range(
        inedge_iterator(inedges_.begin()),
        inedge_iterator(inedges_.end()));
  }

  inline void
  divert_inedges(llvm::cfg_node * new_successor)
  {
    if (this == new_successor)
      return;

    while (NumInEdges())
      InEdges().begin()->divert(new_successor);
  }

  void
  remove_inedges();

  bool
  no_predecessor() const noexcept;

  bool
  single_predecessor() const noexcept;

  bool
  no_successor() const noexcept;

  bool
  single_successor() const noexcept;

  bool
  is_branch() const noexcept
  {
    return NumOutEdges() > 1;
  }

  bool
  has_selfloop_edge() const noexcept;

private:
  ControlFlowGraph & cfg_;
  std::vector<std::unique_ptr<ControlFlowGraphEdge>> outedges_;
  std::unordered_set<ControlFlowGraphEdge *> inedges_;

  friend ControlFlowGraphEdge;
};

template<class T>
static inline bool
is(const cfg_node * node) noexcept
{
  static_assert(
      std::is_base_of<cfg_node, T>::value,
      "Template parameter T must be derived from jlm::cfg_node.");

  return dynamic_cast<const T *>(node) != nullptr;
}

}

#endif

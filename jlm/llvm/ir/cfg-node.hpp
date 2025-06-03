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
class cfg;
class cfg_node;

class cfg_edge final
{
public:
  ~cfg_edge() noexcept
  {}

  cfg_edge(cfg_node * source, cfg_node * sink, size_t index) noexcept;

  cfg_edge(const cfg_edge & other) = delete;
  cfg_edge(cfg_edge && other) = default;

  cfg_edge &
  operator=(const cfg_edge & other) = delete;
  cfg_edge &
  operator=(cfg_edge && other) = default;

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
  using inedge_iterator =
      util::PtrIterator<cfg_edge, std::unordered_set<cfg_edge *>::const_iterator>;
  using inedge_iterator_range = util::IteratorRange<inedge_iterator>;

  using outedge_iterator =
      util::PtrIterator<cfg_edge, std::vector<std::unique_ptr<cfg_edge>>::const_iterator>;
  using outedge_iterator_range = util::IteratorRange<outedge_iterator>;

public:
  virtual ~cfg_node();

protected:
  inline cfg_node(llvm::cfg & cfg)
      : cfg_(cfg)
  {}

public:
  llvm::cfg &
  cfg() const noexcept
  {
    return cfg_;
  }

  size_t
  NumOutEdges() const noexcept;

  cfg_edge *
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

  cfg_edge *
  add_outedge(cfg_node * sink)
  {
    outedges_.push_back(std::make_unique<cfg_edge>(this, sink, NumOutEdges()));
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
  llvm::cfg & cfg_;
  std::vector<std::unique_ptr<cfg_edge>> outedges_;
  std::unordered_set<cfg_edge *> inedges_;

  friend cfg_edge;
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

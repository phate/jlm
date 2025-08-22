/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_AGGREGATION_HPP
#define JLM_LLVM_IR_AGGREGATION_HPP

#include <jlm/llvm/ir/basic-block.hpp>

#include <memory>

namespace jlm::llvm
{

class ControlFlowGraph;

class AggregationNode
{
  using iterator =
      util::PtrIterator<AggregationNode, std::vector<std::unique_ptr<AggregationNode>>::iterator>;
  using const_iterator = util::
      PtrIterator<AggregationNode, std::vector<std::unique_ptr<AggregationNode>>::const_iterator>;

public:
  virtual ~AggregationNode() noexcept;

  AggregationNode()
      : index_(0),
        parent_(nullptr)
  {}

  AggregationNode(const AggregationNode & other) = delete;

  AggregationNode(AggregationNode && other) = delete;

  AggregationNode &
  operator=(const AggregationNode & other) = delete;

  AggregationNode &
  operator=(AggregationNode && other) = delete;

  inline iterator
  begin() noexcept
  {
    return iterator(children_.begin());
  }

  inline const_iterator
  begin() const noexcept
  {
    return const_iterator(children_.begin());
  }

  inline iterator
  end() noexcept
  {
    return iterator(children_.end());
  }

  inline const_iterator
  end() const noexcept
  {
    return const_iterator(children_.end());
  }

  inline size_t
  nchildren() const noexcept
  {
    return children_.size();
  }

  inline void
  add_child(std::unique_ptr<AggregationNode> child)
  {
    size_t index = nchildren();
    children_.emplace_back(std::move(child));
    children_[index]->parent_ = this;
    children_[index]->index_ = index;
  }

  inline AggregationNode *
  child(size_t n) const noexcept
  {
    JLM_ASSERT(n < nchildren());
    return children_[n].get();
  }

  AggregationNode *
  parent() noexcept
  {
    JLM_ASSERT(parent_->child(index_) == this);
    return parent_;
  }

  const AggregationNode *
  parent() const noexcept
  {
    JLM_ASSERT(parent_->child(index_) == this);
    return parent_;
  }

  size_t
  index() const noexcept
  {
    JLM_ASSERT(parent()->child(index_) == this);
    return index_;
  }

  /**
   * Return the number of nodes of the entire subtree.
   */
  size_t
  nnodes() const noexcept
  {
    size_t n = 1;
    for (auto & child : children_)
      n += child->nnodes();

    return n;
  }

  virtual std::string
  debug_string() const = 0;

  /** Normalizes an aggregation tree

  This function normalizes an aggregation tree by reducing nested linear nodes to a single linear
  node. For example, the tree:

  linear
  - linear
  -- block
  -- block
  - block

  is reduced to:

  linear
  - block
  - block
  - block

  */
  static void
  normalize(AggregationNode & node);

private:
  void
  remove_children()
  {
    children_.clear();
  }

  size_t index_;
  AggregationNode * parent_;
  std::vector<std::unique_ptr<AggregationNode>> children_;
};

template<class T>
static inline bool
is(const AggregationNode * node)
{
  static_assert(
      std::is_base_of<AggregationNode, T>::value,
      "Template parameter T must be derived from jlm::AggregationNode");

  return dynamic_cast<const T *>(node) != nullptr;
}

class EntryAggregationNode final : public AggregationNode
{
  class constiterator;

public:
  ~EntryAggregationNode() noexcept override;

  explicit EntryAggregationNode(const std::vector<llvm::argument *> & arguments)
      : arguments_(arguments)
  {}

  constiterator
  begin() const;

  constiterator
  end() const;

  const llvm::argument *
  argument(size_t index) const noexcept
  {
    JLM_ASSERT(index < narguments());
    return arguments_[index];
  }

  size_t
  narguments() const noexcept
  {
    return arguments_.size();
  }

  [[nodiscard]] std::string
  debug_string() const override;

  static std::unique_ptr<AggregationNode>
  create(const std::vector<llvm::argument *> & arguments)
  {
    return std::make_unique<EntryAggregationNode>(arguments);
  }

private:
  std::vector<llvm::argument *> arguments_;
};

class EntryAggregationNode::constiterator final
{
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = const llvm::argument *;
  using difference_type = std::ptrdiff_t;
  using pointer = const llvm::argument **;
  using reference = const llvm::argument *&;

  constexpr constiterator(const std::vector<llvm::argument *>::const_iterator & it)
      : it_(it)
  {}

  const llvm::argument &
  operator*() const
  {
    return *operator->();
  }

  const llvm::argument *
  operator->() const
  {
    return *it_;
  }

  constiterator &
  operator++()
  {
    it_++;
    return *this;
  }

  constiterator
  operator++(int)
  {
    constiterator tmp = *this;
    it_++;
    return tmp;
  }

  bool
  operator==(const constiterator & other) const
  {
    return it_ == other.it_;
  }

  bool
  operator!=(const constiterator & other) const
  {
    return !operator==(other);
  }

private:
  std::vector<llvm::argument *>::const_iterator it_;
};

class ExitAggregationNode final : public AggregationNode
{
  typedef std::vector<const Variable *>::const_iterator const_iterator;

public:
  ~ExitAggregationNode() noexcept override;

  explicit ExitAggregationNode(const std::vector<const Variable *> & results)
      : results_(results)
  {}

  const_iterator
  begin() const
  {
    return results_.begin();
  }

  const_iterator
  end() const
  {
    return results_.end();
  }

  const Variable *
  result(size_t index) const noexcept
  {
    JLM_ASSERT(index < nresults());
    return results_[index];
  }

  size_t
  nresults() const noexcept
  {
    return results_.size();
  }

  [[nodiscard]] std::string
  debug_string() const override;

  static inline std::unique_ptr<AggregationNode>
  create(const std::vector<const Variable *> & results)
  {
    return std::make_unique<ExitAggregationNode>(results);
  }

private:
  std::vector<const Variable *> results_;
};

class BasicBlockAggregationNode final : public AggregationNode
{
public:
  ~BasicBlockAggregationNode() noexcept override;

  BasicBlockAggregationNode() = default;

  explicit BasicBlockAggregationNode(ThreeAddressCodeList && bb)
      : bb_(std::move(bb))
  {}

  [[nodiscard]] std::string
  debug_string() const override;

  const ThreeAddressCodeList &
  tacs() const noexcept
  {
    return bb_;
  }

  static std::unique_ptr<AggregationNode>
  create()
  {
    return std::make_unique<BasicBlockAggregationNode>();
  }

  static std::unique_ptr<AggregationNode>
  create(ThreeAddressCodeList && bb)
  {
    return std::make_unique<BasicBlockAggregationNode>(std::move(bb));
  }

private:
  ThreeAddressCodeList bb_;
};

class LinearAggregationNode final : public AggregationNode
{
public:
  ~LinearAggregationNode() noexcept override;

  LinearAggregationNode(std::unique_ptr<AggregationNode> n1, std::unique_ptr<AggregationNode> n2)
  {
    add_child(std::move(n1));
    add_child(std::move(n2));
  }

  [[nodiscard]] std::string
  debug_string() const override;

  static std::unique_ptr<AggregationNode>
  create(std::unique_ptr<AggregationNode> n1, std::unique_ptr<AggregationNode> n2)
  {
    return std::make_unique<LinearAggregationNode>(std::move(n1), std::move(n2));
  }
};

class BranchAggregationNode final : public AggregationNode
{
public:
  ~BranchAggregationNode() noexcept override;

  BranchAggregationNode() = default;

  [[nodiscard]] std::string
  debug_string() const override;

  static inline std::unique_ptr<AggregationNode>
  create()
  {
    return std::make_unique<BranchAggregationNode>();
  }
};

class LoopAggregationNode final : public AggregationNode
{
public:
  ~LoopAggregationNode() noexcept override;

  explicit LoopAggregationNode(std::unique_ptr<AggregationNode> body)
  {
    add_child(std::move(body));
  }

  [[nodiscard]] std::string
  debug_string() const override;

  static inline std::unique_ptr<AggregationNode>
  create(std::unique_ptr<AggregationNode> body)
  {
    return std::make_unique<LoopAggregationNode>(std::move(body));
  }
};

/** \brief Aggregate a properly structured CFG to a aggregation tree.
 *
 * This function reduces a properly structured CFG to an aggregation tree. The CFG is only
 * allowed to consist of the following subgraphs:
 *
 * 1. Linear subgraphs, such as:
 * \dot
 * 	digraph linear {
 * 		A -> B;
 * 	}
 * \enddot
 *
 * 2. Branch subgraphs, such as:
 * \dot
 *   digraph branches {
 * 		Split -> A;
 * 		Split -> B;
 * 		Split -> C;
 * 		A -> Join;
 * 		B -> Join;
 * 		C -> Join;
 * 	}
 * \enddot
 *
 * 3. Tail-Controlled Loops, such as:
 * \dot
 * 	digraph tcloop {
 * 		A -> Loop;
 * 		Loop -> Loop;
 * 		Loop -> B;
 * 	}
 * \enddot
 *
 * These subgraphs can be arbitrarily nested. Please refer to Reissmann et al. - RVSDG: An
 * Intermediate Representation for Optimizing Compilers [https://doi.org/10.1145/3391902] for more
 * information.
 */
std::unique_ptr<AggregationNode>
aggregate(ControlFlowGraph & cfg);

size_t
ntacs(const AggregationNode & root);

}

#endif

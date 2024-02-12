/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */
#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_WORKLIST_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_WORKLIST_HPP

#include <jlm/util/common.hpp>

#include <limits>
#include <queue>
#include <stack>
#include <vector>

namespace jlm::llvm::aa
{

/**
 * Class for managing a set of work items, that are waiting to be visited by an algorithm.
 * @tparam T the integer type used to index work items.
 */
template<typename T>
class Worklist
{
public:
  Worklist()
  {}

  Worklist(const Worklist & other) = delete;
  Worklist(Worklist && other) = delete;
  Worklist &
  operator=(const Worklist & other) = delete;
  Worklist &
  operator=(Worklist && other) = delete;

  virtual ~Worklist() = default;

  /**
   * @return true if there are work items left to be visited
   */
  virtual bool
  HasWorkItem() = 0;

  /**
   * Removes one work item from the worklist.
   * Requires there to be at least one work item left.
   * @return the removed work item.
   */
  virtual T
  PopWorkItem() = 0;

  /**
   * Adds a work item to the worklist.
   * If the item is already present, the item is not added again, but its position may be changed.
   * @param item the work item to be added.
   */
  virtual void
  PushWorkItem(T item) = 0;
};

/**
 * Class for managing a set of work items using a stack.
 * Pushing a work item already on the stack is a no-op.
 * @tparam T the integer type used to index work items.
 */
template<typename T>
class LIFOWorklist final : public Worklist<T>
{
public:
  explicit LIFOWorklist(T size)
      : OnList_(size, false)
  {}

  ~LIFOWorklist() override = default;

  bool
  HasWorkItem() override
  {
    return !WorkItems_.empty();
  }

  T
  PopWorkItem() override
  {
    JLM_ASSERT(HasWorkItem());
    T item = WorkItems_.top();
    WorkItems_.pop();
    OnList_[item] = false;
    return item;
  }

  void
  PushWorkItem(T item) override
  {
    JLM_ASSERT(0 <= item < OnList_.size());
    if (OnList_[item])
      return;
    OnList_[item] = true;
    WorkItems_.push(item);
  }

private:
  // Tracking which work items are already on the list
  std::vector<bool> OnList_;

  // The stack used to order the work items
  std::stack<T> WorkItems_;
};

/**
 * Class for managing a set of work items using a queue.
 * Pushing a work item already in the queue is a no-op.
 * @tparam T the integer type used to index work items.
 */
template<typename T>
class FIFOWorklist final : public Worklist<T>
{
public:
  explicit FIFOWorklist(T size)
      : OnList_(size, false)
  {}

  ~FIFOWorklist() override = default;

  bool
  HasWorkItem() override
  {
    return !WorkItems_.empty();
  }

  T
  PopWorkItem() override
  {
    JLM_ASSERT(HasWorkItem());
    T item = WorkItems_.front();
    WorkItems_.pop();
    OnList_[item] = false;
    return item;
  }

  void
  PushWorkItem(T item) override
  {
    JLM_ASSERT(0 <= item < OnList_.size());
    if (OnList_[item])
      return;
    OnList_[item] = true;
    WorkItems_.push(item);
  }

private:
  // Tracking which work items are already on the list
  std::vector<bool> OnList_;

  // The queue used to order the items
  std::queue<T> WorkItems_;
};

/**
 * Class for managing a set of work items using a priority queue.
 * The next work item to fire is the one that was least recently fired (LRF)
 * @tparam T the integer type used to index work items.
 */
template<typename T>
class LRFWorklist final : public Worklist<T>
{
public:
  explicit LRFWorklist(T size)
      : FireCounter_(0), LastFire_(size, 0)
  {}

  ~LRFWorklist() override = default;

  bool
  HasWorkItem() override
  {
    return !WorkItems_.empty();
  }

  T
  PopWorkItem() override
  {
    JLM_ASSERT(HasWorkItem());
    auto [_, item] = WorkItems_.top();
    WorkItems_.pop();
    // Note down the moment this item fired
    LastFire_[item] = ++FireCounter_;
    return item;
  }

  void
  PushWorkItem(T item) override
  {
    JLM_ASSERT(0 <= item < LastFire_.size());
    if (LastFire_[item] == InQueueSentinelValue)
      return;
    // Add the work item to the priority queue based on when it was last fired
    WorkItems_.push({LastFire_[item], item});
    LastFire_[item] = InQueueSentinelValue;
  }

private:
  // If a work item is currently in the queue, its "last fire" is set to this value
  static inline const size_t InQueueSentinelValue = std::numeric_limits<size_t>::max();

  // The type of a priority queue giving the highest priority to the lowest value
  template<typename U>
  using MinPriorityQueue = std::priority_queue<U, std::vector<U>, std::greater<U>>;

  // The priority queue used to order the items
  MinPriorityQueue<std::pair<size_t, T>> WorkItems_;

  // Counter used to order fire events (work items being popped)
  size_t FireCounter_;

  // For each work item, when the item was last fired.
  // If the work item is currently in the queue, InQueueSentinelValue is used instead
  std::vector<size_t> LastFire_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_WORKLIST_HPP

/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */
#ifndef JLM_UTIL_WORKLIST_HPP
#define JLM_UTIL_WORKLIST_HPP

#include "common.hpp"
#include "HashSet.hpp"

#include <algorithm>
#include <limits>
#include <queue>
#include <stack>
#include <unordered_map>
#include <vector>

namespace jlm::util
{

/**
 * Class for managing a set of work items, that are waiting to be visited by an algorithm.
 * The implementation decides in what order remaining work items should be processed.
 * @tparam T the type of the work items.
 */
template<typename T>
class Worklist
{
public:
  virtual ~Worklist() = default;

  Worklist() = default;

  Worklist(const Worklist & other) = delete;

  Worklist(Worklist && other) = default;

  Worklist &
  operator=(const Worklist & other) = delete;

  Worklist &
  operator=(Worklist && other) = default;

  /**
   * @return true if there are work items left to be visited
   */
  [[nodiscard]] virtual bool
  HasMoreWorkItems() const noexcept = 0;

  /**
   * Removes one work item from the worklist.
   * Requires there to be at least one work item left.
   * @return the removed work item.
   */
  [[nodiscard]] virtual T
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
 * Worklist implementation using a stack.
 * Pushing a work item that is already on the stack is a no-op.
 * @tparam T the type of the work items.
 * @see Worklist
 */
template<typename T>
class LifoWorklist final : public Worklist<T>
{
public:
  ~LifoWorklist() override = default;

  LifoWorklist() = default;

  [[nodiscard]] bool
  HasMoreWorkItems() const noexcept override
  {
    return !WorkItems_.empty();
  }

  T
  PopWorkItem() override
  {
    JLM_ASSERT(HasMoreWorkItems());
    T item = WorkItems_.top();
    WorkItems_.pop();
    OnList_.Remove(item);
    return item;
  }

  void
  PushWorkItem(T item) override
  {
    if (OnList_.Insert(item))
      WorkItems_.push(item);
  }

private:
  // Tracking which work items are already on the list
  util::HashSet<T> OnList_;

  // The stack used to order the work items
  std::stack<T> WorkItems_;
};

/**
 * Worklist implementation using a queue.
 * Pushing a work item that is already in the queue is a no-op.
 * @tparam T the type of the work items.
 * @see Worklist
 */
template<typename T>
class FifoWorklist final : public Worklist<T>
{
public:
  ~FifoWorklist() override = default;

  FifoWorklist() = default;

  [[nodiscard]] bool
  HasMoreWorkItems() const noexcept override
  {
    return !WorkItems_.empty();
  }

  T
  PopWorkItem() override
  {
    JLM_ASSERT(HasMoreWorkItems());
    T item = WorkItems_.front();
    WorkItems_.pop();
    OnList_.Remove(item);
    return item;
  }

  void
  PushWorkItem(T item) override
  {
    if (OnList_.Insert(item))
      WorkItems_.push(item);
  }

private:
  // Tracking which work items are already on the list
  util::HashSet<T> OnList_;

  // The queue used to order the items
  std::queue<T> WorkItems_;
};

/**
 * Worklist implementation using a priority queue, ordering work items by "Least Recently Fired".
 * Each work item is time stamped when it leaves the work list. When selecting a work item from
 * the list, the item with the oldest time stemp (or no time stamp, if any) is chosen.
 * LRF is presented in
 *   A. Kanamori and D. Weise "Worklist management strategies for Dataflow Analysis" (1994),
 * and used in
 *   Pierce's "Online cycle detection and difference propagation for pointer analysis" (2003).
 * @tparam T the type of the work items.
 * @see Worklist
 */
template<typename T>
class LrfWorklist final : public Worklist<T>
{
public:
  ~LrfWorklist() override = default;

  LrfWorklist() = default;

  [[nodiscard]] bool
  HasMoreWorkItems() const noexcept override
  {
    return !WorkItems_.empty();
  }

  T
  PopWorkItem() override
  {
    JLM_ASSERT(HasMoreWorkItems());
    auto [_, item] = WorkItems_.top();
    WorkItems_.pop();
    // Note down the moment this item fired
    LastFire_[item] = ++FireCounter_;
    return item;
  }

  void
  PushWorkItem(T item) override
  {
    size_t & lastFire = LastFire_[item];
    if (lastFire == InQueueSentinelValue)
      return;

    // Add the work item to the priority queue based on when it was last fired
    WorkItems_.push({ lastFire, item });
    lastFire = InQueueSentinelValue;
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
  size_t FireCounter_ = 0;

  // For each work item, when the item was last fired.
  // If the work item is currently in the queue, InQueueSentinelValue is used instead
  std::unordered_map<T, size_t> LastFire_;
};

/**
 * Worklist implementation similar to "Least Recently Fired", but using two lists.
 * Pushed work items are added to `next`, and popped from `current`.
 * When `current` is empty, `next` is sorted by last fired time, and becomes the new `current`.
 * Pushing a work item that is already in next or current is a no-op.
 * Two-phase LRF is presented by
 *   B. Hardekopf and C. Lin "The And and the Grasshopper: Fast and Accurate Pointer Analysis
 *   for Millions of Lines of Code" (2007)
 * @tparam T the type of the work items.
 * @see Worklist
 */
template<typename T>
class TwoPhaseLrfWorklist final : public Worklist<T>
{
public:
  ~TwoPhaseLrfWorklist() override = default;

  TwoPhaseLrfWorklist() = default;

  [[nodiscard]] bool
  HasMoreWorkItems() const noexcept override
  {
    return !Current_.empty() || !Next_.empty();
  }

  T
  PopWorkItem() override
  {
    if (Current_.empty())
    {
      JLM_ASSERT(!Next_.empty());
      std::swap(Current_, Next_);
      std::sort(Current_.rbegin(), Current_.rend()); // The least recently first work item gets last
    }

    auto [_, item] = Current_.back();
    Current_.pop_back();
    // Note down the moment this item fired
    LastFire_[item] = ++FireCounter_;
    return item;
  }

  void
  PushWorkItem(T item) override
  {
    size_t & lastFire = LastFire_[item];
    if (lastFire == InQueueSentinelValue)
      return;

    // Add the work item to the next list, with its last fire time used for sorting
    Next_.push_back({ lastFire, item });
    lastFire = InQueueSentinelValue;
  }

private:
  // If a work item is currently in the queue, its "last fire" is set to this value
  static inline const size_t InQueueSentinelValue = std::numeric_limits<size_t>::max();

  // The current list being popped. The pair's first is when the work item was last fired
  std::vector<std::pair<size_t, T>> Current_;

  // The next list. The pair's first is when the work item was last fired
  // When current is empty, this list is sorted by least recently fired, and becomes the new current
  std::vector<std::pair<size_t, T>> Next_;

  // Counter used to order fire events (work items being popped)
  size_t FireCounter_ = 0;

  // For each work item, when the item was last fired.
  // If the work item is currently in the queue, InQueueSentinelValue is used instead
  std::unordered_map<T, size_t> LastFire_;
};

/**
 * A fake worklist that remembers which work items have been pushed,
 * but without providing any kind of iteration interface for accessing them.
 * Each work item must be explicitly removed by name.
 * Used to implement the Topological worklist policy, which is not technically a worklist policy.
 * @tparam T the type of the work items.
 * @see Worklist
 */
template<typename T>
class Workset final : public Worklist<T>
{
public:
  ~Workset() override = default;

  Workset() = default;

  [[nodiscard]] bool
  HasMoreWorkItems() const noexcept override
  {
    return !PushedItems_.IsEmpty();
  }

  T
  PopWorkItem() override
  {
    JLM_UNREACHABLE("The Workset does not provide an iteration order");
  }

  void
  PushWorkItem(T item) override
  {
    PushedItems_.Insert(item);
  }

  [[nodiscard]] bool
  HasWorkItem(T item) const noexcept
  {
    return PushedItems_.Contains(item);
  }

  void
  RemoveWorkItem(T item)
  {
    PushedItems_.Remove(item);
  }

private:
  util::HashSet<T> PushedItems_;
};

}

#endif // JLM_UTIL_WORKLIST_HPP

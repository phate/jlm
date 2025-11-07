/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/ModRefSummary.hpp>

namespace jlm::llvm::aa
{
void
MemoryNodeIntervalSet::sortAndCompact()
{
  if (intervals_.empty())
    return;

  std::sort(intervals_.begin(), intervals_.end());
  size_t currentInterval = 0;

  // Check if intervals can be merged into currentInterval instead of being new intervals
  for (size_t i = 1; i < intervals_.size(); i++)
  {
    if (intervals_[currentInterval].end >= intervals_[i].start)
    {
      // The intervals are mergeable!
      intervals_[currentInterval].end =
          std::max(intervals_[currentInterval].end, intervals_[i].end);
    }
    else
    {
      // The i interval can not be merged with currentInterval

      // If currentInterval is a non-empty interval, keep it by incrementing currentInterval
      if (intervals_[currentInterval].start < intervals_[currentInterval].end)
        currentInterval++;

      // make the new currentInterval be intervals_[i]
      if (currentInterval != i)
        intervals_[currentInterval] = intervals_[i];
    }
  }

  // Keep the final currentInterval, if it is non-empty
  if (intervals_[currentInterval].start < intervals_[currentInterval].end)
    currentInterval++;

  // Discard any intervals after the final interval
  intervals_.resize(currentInterval);
}

MemoryNodeIntervalSetIterator::MemoryNodeIntervalSetIterator(const MemoryNodeIntervalSet & set)
    : set_(set),
      index_(0)
{}

std::optional<MemoryNodeInterval>
MemoryNodeIntervalSetIterator::peek() const
{
  if (index_ < set_.getIntervals().size())
    return set_.getIntervals()[index_];
  return std::nullopt;
}

void
MemoryNodeIntervalSetIterator::next()
{
  index_++;
}

MemoryNodeIntervalSetUnionIterator::MemoryNodeIntervalSetUnionIterator(
    MemoryNodeIntervalSetIterator a,
    MemoryNodeIntervalSetIterator b)
    : a_(a),
      b_(b)
{
  // initialize the first interval
  next();
}

std::optional<MemoryNodeInterval>
MemoryNodeIntervalSetUnionIterator::peek() const
{
  return current_;
}

void
MemoryNodeIntervalSetUnionIterator::next()
{
  MemoryNodeOrderingIndex start = std::numeric_limits<MemoryNodeOrderingIndex>::max();
  MemoryNodeOrderingIndex end = 0;

  // Pick the interval that starts the earliest
  if (const auto a = a_.peek(); a && a->start < start)
  {
    start = a->start;
    end = a->end;
  }
  if (const auto b = a_.peek(); b && b->start < start)
  {
    start = b->start;
    end = b->end;
  }
  if (start == std::numeric_limits<MemoryNodeOrderingIndex>::max())
  {
    // If there are no more intervals left, we are done
    current_ = std::nullopt;
    return;
  }

  // Consume all intervals that are within [start, end)
  // If intervals are overlapping or adjacent, move end back to make one large interval.
  while (true)
  {
    if (const auto a = a_.peek(); a && a->start <= end)
    {
      end = std::max(end, a->end);
      a_.next();
    }
    else if (const auto b = b_.peek(); b && b->start <= end)
    {
      end = std::max(end, b->end);
      b_.next();
    }
    else
    {
      break;
    }
  }

  current_ = MemoryNodeInterval(start, end);
}

MemoryNodeIntervalSetDifferenceIterator::MemoryNodeIntervalSetDifferenceIterator(
    MemoryNodeIntervalSetIterator loads,
    MemoryNodeIntervalSetIterator stores)
    : loads_(loads),
      stores_(stores),
      lastEnd_(0)
{
  // initialize the first interval
  next();
}

std::optional<std::pair<MemoryNodeInterval, bool>>
MemoryNodeIntervalSetDifferenceIterator::peek() const
{
  return current_;
}

void
MemoryNodeIntervalSetDifferenceIterator::next()
{
  MemoryNodeOrderingIndex start = std::numeric_limits<MemoryNodeOrderingIndex>::max();
  MemoryNodeOrderingIndex end = 0;
  bool isStore = false;

  // Use whichever interval comes first. If they both start at the same index, use the store
  if (const auto store = stores_.peek())
  {
    const auto storeStart = std::max(lastEnd_, store->start);
    if (storeStart < start)
    {
      start = storeStart;
      end = store->end;
      isStore = true;
    }
  }
  if (const auto load = loads_.peek())
  {
    const auto loadStart = std::max(lastEnd_, load->start);
    if (loadStart < start)
    {
      start = loadStart;
      end = load->end;
      isStore = false;
    }
  }
  if (start == std::numeric_limits<MemoryNodeOrderingIndex>::max())
  {
    current_ = std::nullopt;
    return;
  }

  // Consume intervals from the load and store interval streams if they are within [start, end)
  // Partially overlapping or adjacent intervals can be used to expand end
  // We can only consume store intervals if isStore is true
  while (true)
  {
    if (const auto store = stores_.peek(); store && store->start <= end)
    {
      if (isStore)
      {
        // If the interval we are working on is a store, consume this store and possibly expand it
        end = std::max(end, store->end);
        stores_.next();
      }
      else
      {
        // If we are working on a load, cut of the load interval before the store starts
        // We avoid popping the load on purpose, in case of situations like
        //          [ store ]
        //     [      load       ]
        // Here we want separate parts of the load interval to be returned twice.
        end = store->start;
        break;
      }
    }
    else if (const auto load = loads_.peek(); load && load->start <= end)
    {
      if (!isStore)
      {
        // If we are working on a load interval, we can extend the end if there is partial overlap
        end = std::max(end, load->end);
        loads_.next();
      }
      else if (load->end <= end)
      {
        // If the load interval is fully within the store interval, we can also consume it
        loads_.next();
      }
      else
      {
        // If we get here, we are unable to extend the (start, end] store interval any longer,
        // and can not consume the next load interval either
        break;
      }
    }
    else
    {
      break;
    }

  }

  // Keep track of where we ended, to ensure the next interval starts at-or-after it
  lastEnd_ = end;
  current_ = { MemoryNodeInterval(start, end), isStore };
}

}

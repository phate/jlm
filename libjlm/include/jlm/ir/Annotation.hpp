/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_ANNOTATION_HPP
#define JLM_IR_ANNOTATION_HPP

#include <jlm/common.hpp>
#include <jlm/util/iterator_range.hpp>

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace jlm {

class aggnode;
class variable;

class VariableSet final {

  class ConstIterator final : public std::iterator<std::forward_iterator_tag, const jlm::variable*, ptrdiff_t> {
  public:
    explicit
    ConstIterator(const std::unordered_set<const jlm::variable*>::const_iterator & it)
      : It_(it)
    {}

  public:
    const jlm::variable &
    GetVariable() const noexcept
    {
      return **It_;
    }

    const jlm::variable &
    operator*() const
    {
      return GetVariable();
    }

    const jlm::variable *
    operator->() const
    {
      return &GetVariable();
    }

    ConstIterator &
    operator++()
    {
      ++It_;
      return *this;
    }

    ConstIterator
    operator++(int)
    {
      ConstIterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool
    operator==(const ConstIterator & other) const
    {
      return It_ == other.It_;
    }

    bool
    operator!=(const ConstIterator & other) const
    {
      return !operator==(other);
    }

  private:
    std::unordered_set<const jlm::variable*>::const_iterator It_;
  };

  using ConstRange = iterator_range<ConstIterator>;

public:
  VariableSet()
  = default;

  VariableSet(std::initializer_list<const variable*> init)
  : Set_(init)
  {}

  ConstRange
  Variables() const noexcept
  {
    return {ConstIterator(Set_.begin()), ConstIterator(Set_.end())};
  }

	bool
	Contains(const variable & v) const
	{
		return Set_.find(&v) != Set_.end();
	}

  bool
  Contains(const VariableSet & variableSet) const
  {
    if (variableSet.Size() > Size())
      return false;

    return std::all_of(
      variableSet.Set_.begin(),
      variableSet.Set_.end(),
      [&](const variable * v){ return Contains(*v); });
  }

	size_t
	Size() const noexcept
	{
		return Set_.size();
	}

	void
	Insert(const variable & v)
	{
		Set_.insert(&v);
	}

	void
	Insert(const VariableSet & variableSet)
	{
		Set_.insert(variableSet.Set_.begin(), variableSet.Set_.end());
	}

	void
	Remove(const variable & v)
	{
		Set_.erase(&v);
	}

	void
	Remove(const VariableSet & variableSet)
	{
		for (auto & v : variableSet.Variables())
			Remove(v);
	}

	void
	Intersect(const VariableSet & variableSet)
	{
    for (auto it = Set_.begin(); it != Set_.end();) {
      if (!variableSet.Contains(**it))
        it = Set_.erase(it);
      else
        it++;
    }
	}

	bool
	operator==(const VariableSet & other) const
	{
		if (Size() != other.Size())
			return false;

    return std::all_of(
      Set_.begin(),
      Set_.end(),
      [&](const variable * v){ return Contains(*v); });
	}

	bool
	operator!=(const VariableSet & other) const
	{
		return !(*this == other);
	}

  std::string
  DebugString() const noexcept;

private:
	std::unordered_set<const variable*> Set_;
};

class DemandSet {
public:
	virtual
	~DemandSet() noexcept;

	DemandSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
    : ReadSet_(std::move(readSet))
    , AllWriteSet_(std::move(allWriteSet))
    , FullWriteSet_(std::move(fullWriteSet))
  {}

  DemandSet(const DemandSet&) = delete;

  DemandSet(DemandSet&&) noexcept = delete;

  DemandSet&
  operator=(const DemandSet&) = delete;

  DemandSet&
  operator=(DemandSet&&) = delete;

  const VariableSet &
  ReadSet() const noexcept
  {
    return ReadSet_;
  }

  const VariableSet &
  AllWriteSet() const noexcept
  {
    return AllWriteSet_;
  }

  const VariableSet &
  FullWriteSet() const noexcept
  {
    return FullWriteSet_;
  }

  virtual bool
  operator==(const DemandSet & other)
  {
    return ReadSet_ == other.ReadSet_
        && AllWriteSet_ == other.AllWriteSet_
        && FullWriteSet_ == other.FullWriteSet_;
  }

  bool
  operator!=(const DemandSet & other)
  {
    return !(*this == other);
  }

  virtual std::string
  DebugString() const noexcept = 0;

private:
	VariableSet ReadSet_;
	VariableSet AllWriteSet_;
	VariableSet FullWriteSet_;
};

class EntryDemandSet final : public DemandSet {
public:
  ~EntryDemandSet() noexcept override;

  EntryDemandSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
  : DemandSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  {}

  EntryDemandSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet,
    VariableSet topSet)
  : DemandSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  , TopSet_(std::move(topSet))
  {}

  static std::unique_ptr<EntryDemandSet>
  Create(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
  {
    return std::make_unique<EntryDemandSet>(
      std::move(readSet),
      std::move(allWriteSet),
      std::move(fullWriteSet));
  }

  std::string
  DebugString() const noexcept override;

  bool
  operator==(const DemandSet & other) override;

  VariableSet TopSet_;
};

class ExitDemandSet final : public DemandSet {
public:
  ~ExitDemandSet() noexcept override;

  ExitDemandSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
    : DemandSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  {}

  static std::unique_ptr<ExitDemandSet>
  Create(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
  {
    return std::make_unique<ExitDemandSet>(
      std::move(readSet),
      std::move(allWriteSet),
      std::move(fullWriteSet));
  }

  std::string
  DebugString() const noexcept override;

  bool
  operator==(const DemandSet & other) override;
};

class BasicBlockDemandSet final : public DemandSet {
public:
  ~BasicBlockDemandSet() noexcept override;

  BasicBlockDemandSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
    : DemandSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  {}

  static std::unique_ptr<BasicBlockDemandSet>
  Create(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
  {
    return std::make_unique<BasicBlockDemandSet>(
      std::move(readSet),
      std::move(allWriteSet),
      std::move(fullWriteSet));
  }

  std::string
  DebugString() const noexcept override;

  bool
  operator==(const DemandSet & other) override;
};

class LinearDemandSet final : public DemandSet {
public:
  ~LinearDemandSet() noexcept override;

  LinearDemandSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
    : DemandSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  {}

  LinearDemandSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet,
    VariableSet topSet,
    VariableSet bottomSet)
  : DemandSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  , TopSet_(std::move(topSet))
  , BottomSet_(std::move(bottomSet))
  {}

  static std::unique_ptr<LinearDemandSet>
  Create(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
  {
    return std::make_unique<LinearDemandSet>(
      std::move(readSet),
      std::move(allWriteSet),
      std::move(fullWriteSet));
  }

  std::string
  DebugString() const noexcept override;

  bool
  operator==(const DemandSet & other) override;

  VariableSet TopSet_;
  VariableSet BottomSet_;
};

class BranchDemandSet final : public DemandSet {
public:
  ~BranchDemandSet() noexcept override;

  BranchDemandSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
    : DemandSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  {}

  BranchDemandSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet,
    VariableSet topSet,
    VariableSet bottomSet)
  : DemandSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  , TopSet_(std::move(topSet))
  , BottomSet_(std::move(bottomSet))
  {}

  static std::unique_ptr<BranchDemandSet>
  Create(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
  {
    return std::make_unique<BranchDemandSet>(
      std::move(readSet),
      std::move(allWriteSet),
      std::move(fullWriteSet));
  }

  std::string
  DebugString() const noexcept override;

  bool
  operator==(const DemandSet & other) override;

  VariableSet TopSet_;
  VariableSet BottomSet_;
};

class LoopDemandSet final : public DemandSet {
public:
  ~LoopDemandSet() noexcept override;

  LoopDemandSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
    : DemandSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  {}

  LoopDemandSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet,
    VariableSet loopVariables)
  : DemandSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  , LoopVariables_(std::move(loopVariables))
  {}

  static std::unique_ptr<LoopDemandSet>
  Create(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
  {
    return std::make_unique<LoopDemandSet>(
      std::move(readSet),
      std::move(allWriteSet),
      std::move(fullWriteSet));
  }

  std::string
  DebugString() const noexcept override;

  bool
  operator==(const DemandSet & other) override;

  const VariableSet &
  LoopVariables() const noexcept
  {
    return LoopVariables_;
  }

  void
  SetLoopVariables(VariableSet loopVariables) noexcept
  {
    LoopVariables_ = std::move(loopVariables);
  }

private:
  VariableSet LoopVariables_;
};

class DemandMap final {
public:
  DemandMap()
  = default;

  DemandMap(const DemandMap&) = delete;

  DemandMap(DemandMap&&) noexcept = delete;

  DemandMap&
  operator=(const DemandMap&) = delete;

  DemandMap&
  operator=(DemandMap&&) noexcept = delete;

  bool
  Contains(const aggnode & aggregationNode) const noexcept
  {
    return Map_.find(&aggregationNode) != Map_.end();
  }

  template <class T> T&
  Lookup(const aggnode & aggregationNode) const noexcept
  {
    JLM_ASSERT(Contains(aggregationNode));
    auto & demandSet = *Map_.find(&aggregationNode)->second;
    JLM_ASSERT(dynamic_cast<const T*>(&demandSet));
    return *static_cast<T*>(&demandSet);
  }

  void
  Insert(
    const aggnode & aggregationNode,
    std::unique_ptr<DemandSet> demandSet)
  {
    JLM_ASSERT(!Contains(aggregationNode));
    Map_[&aggregationNode] = std::move(demandSet);
  }

  static std::unique_ptr<DemandMap>
  Create()
  {
    return std::make_unique<DemandMap>();
  }

private:
  std::unordered_map<const aggnode*, std::unique_ptr<DemandSet>> Map_;
};

std::unique_ptr<DemandMap>
Annotate(const jlm::aggnode & aggregationTreeRoot);

}

#endif

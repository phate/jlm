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
		std::unordered_set<const variable*> intersect;
		for (auto & v : variableSet.Variables()) {
			if (Contains(v))
				intersect.insert(&v);
		}

		Set_ = intersect;
	}

	void
	Subtract(const VariableSet & variableSet)
	{
		for (auto & v : variableSet.Variables())
			Remove(v);
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

private:
	std::unordered_set<const variable*> Set_;
};

class DemandSet {
public:
	virtual
	~DemandSet();

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


	static inline std::unique_ptr<DemandSet>
	Create(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
	{
		return std::make_unique<DemandSet>(
      std::move(readSet),
      std::move(allWriteSet),
      std::move(fullWriteSet));
	}

	VariableSet top;
	VariableSet bottom;

private:
	VariableSet ReadSet_;
	VariableSet AllWriteSet_;
	VariableSet FullWriteSet_;
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

  DemandSet&
  Lookup(const aggnode & aggregationNode) const noexcept
  {
    JLM_ASSERT(Contains(aggregationNode));
    return *Map_.find(&aggregationNode)->second;
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

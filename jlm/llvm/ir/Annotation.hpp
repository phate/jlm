/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_ANNOTATION_HPP
#define JLM_LLVM_IR_ANNOTATION_HPP

#include <jlm/util/common.hpp>
#include <jlm/util/iterator_range.hpp>

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace jlm {

class aggnode;
class variable;

class VariableSet final {

  class ConstIterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const jlm::variable*;
    using difference_type = std::ptrdiff_t;
    using pointer = const jlm::variable**;
    using reference = const jlm::variable*&;

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

class AnnotationSet {
public:
	virtual
	~AnnotationSet() noexcept;

	AnnotationSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
    : ReadSet_(std::move(readSet))
    , AllWriteSet_(std::move(allWriteSet))
    , FullWriteSet_(std::move(fullWriteSet))
  {}

  AnnotationSet(const AnnotationSet&) = delete;

  AnnotationSet(AnnotationSet&&) noexcept = delete;

  AnnotationSet&
  operator=(const AnnotationSet&) = delete;

  AnnotationSet&
  operator=(AnnotationSet&&) = delete;

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
  operator==(const AnnotationSet & other)
  {
    return ReadSet_ == other.ReadSet_
        && AllWriteSet_ == other.AllWriteSet_
        && FullWriteSet_ == other.FullWriteSet_;
  }

  bool
  operator!=(const AnnotationSet & other)
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

class EntryAnnotationSet final : public AnnotationSet {
public:
  ~EntryAnnotationSet() noexcept override;

  EntryAnnotationSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
  : AnnotationSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  {}

  EntryAnnotationSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet,
    VariableSet topSet)
  : AnnotationSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  , TopSet_(std::move(topSet))
  {}

  static std::unique_ptr<EntryAnnotationSet>
  Create(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
  {
    return std::make_unique<EntryAnnotationSet>(
      std::move(readSet),
      std::move(allWriteSet),
      std::move(fullWriteSet));
  }

  std::string
  DebugString() const noexcept override;

  bool
  operator==(const AnnotationSet & other) override;

  VariableSet TopSet_;
};

class ExitAnnotationSet final : public AnnotationSet {
public:
  ~ExitAnnotationSet() noexcept override;

  ExitAnnotationSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
    : AnnotationSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  {}

  static std::unique_ptr<ExitAnnotationSet>
  Create(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
  {
    return std::make_unique<ExitAnnotationSet>(
      std::move(readSet),
      std::move(allWriteSet),
      std::move(fullWriteSet));
  }

  std::string
  DebugString() const noexcept override;

  bool
  operator==(const AnnotationSet & other) override;
};

class BasicBlockAnnotationSet final : public AnnotationSet {
public:
  ~BasicBlockAnnotationSet() noexcept override;

  BasicBlockAnnotationSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
    : AnnotationSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  {}

  static std::unique_ptr<BasicBlockAnnotationSet>
  Create(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
  {
    return std::make_unique<BasicBlockAnnotationSet>(
      std::move(readSet),
      std::move(allWriteSet),
      std::move(fullWriteSet));
  }

  std::string
  DebugString() const noexcept override;

  bool
  operator==(const AnnotationSet & other) override;
};

class LinearAnnotationSet final : public AnnotationSet {
public:
  ~LinearAnnotationSet() noexcept override;

  LinearAnnotationSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
    : AnnotationSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  {}

  static std::unique_ptr<LinearAnnotationSet>
  Create(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
  {
    return std::make_unique<LinearAnnotationSet>(
      std::move(readSet),
      std::move(allWriteSet),
      std::move(fullWriteSet));
  }

  std::string
  DebugString() const noexcept override;

  bool
  operator==(const AnnotationSet & other) override;
};

class BranchAnnotationSet final : public AnnotationSet {
public:
  ~BranchAnnotationSet() noexcept override;

  BranchAnnotationSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
    : AnnotationSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  {}

  BranchAnnotationSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet,
    VariableSet inputVariables,
    VariableSet outputVariables)
  : AnnotationSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  , InputVariables_(std::move(inputVariables))
  , OutputVariables_(std::move(outputVariables))
  {}

  static std::unique_ptr<BranchAnnotationSet>
  Create(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
  {
    return std::make_unique<BranchAnnotationSet>(
      std::move(readSet),
      std::move(allWriteSet),
      std::move(fullWriteSet));
  }

  std::string
  DebugString() const noexcept override;

  bool
  operator==(const AnnotationSet & other) override;

  const VariableSet &
  InputVariables() const noexcept
  {
    return InputVariables_;
  }

  void
  SetInputVariables(VariableSet inputVariables) noexcept
  {
    InputVariables_ = std::move(inputVariables);
  }

  const VariableSet &
  OutputVariables() const noexcept
  {
    return OutputVariables_;
  }

  void
  SetOutputVariables(VariableSet outputVariables) noexcept
  {
    OutputVariables_ = std::move(outputVariables);
  }

private:
  VariableSet InputVariables_;
  VariableSet OutputVariables_;
};

class LoopAnnotationSet final : public AnnotationSet {
public:
  ~LoopAnnotationSet() noexcept override;

  LoopAnnotationSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
    : AnnotationSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  {}

  LoopAnnotationSet(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet,
    VariableSet loopVariables)
  : AnnotationSet(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet))
  , LoopVariables_(std::move(loopVariables))
  {}

  static std::unique_ptr<LoopAnnotationSet>
  Create(
    VariableSet readSet,
    VariableSet allWriteSet,
    VariableSet fullWriteSet)
  {
    return std::make_unique<LoopAnnotationSet>(
      std::move(readSet),
      std::move(allWriteSet),
      std::move(fullWriteSet));
  }

  std::string
  DebugString() const noexcept override;

  bool
  operator==(const AnnotationSet & other) override;

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

class AnnotationMap final {
public:
  AnnotationMap()
  = default;

  AnnotationMap(const AnnotationMap&) = delete;

  AnnotationMap(AnnotationMap&&) noexcept = delete;

  AnnotationMap&
  operator=(const AnnotationMap&) = delete;

  AnnotationMap&
  operator=(AnnotationMap&&) noexcept = delete;

  bool
  Contains(const aggnode & aggregationNode) const noexcept
  {
    return Map_.find(&aggregationNode) != Map_.end();
  }

  template <class T> T&
  Lookup(const aggnode & aggregationNode) const noexcept
  {
    JLM_ASSERT(Contains(aggregationNode));
    auto & demandSet = Map_.find(&aggregationNode)->second;
    return *AssertedCast<T>(demandSet.get());
  }

  void
  Insert(
    const aggnode & aggregationNode,
    std::unique_ptr<AnnotationSet> annotationSet)
  {
    JLM_ASSERT(!Contains(aggregationNode));
    Map_[&aggregationNode] = std::move(annotationSet);
  }

  static std::unique_ptr<AnnotationMap>
  Create()
  {
    return std::make_unique<AnnotationMap>();
  }

private:
  std::unordered_map<const aggnode*, std::unique_ptr<AnnotationSet>> Map_;
};

std::unique_ptr<AnnotationMap>
Annotate(const jlm::aggnode & aggregationTreeRoot);

}

#endif

/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_CALL_SUMMARY_HPP
#define JLM_LLVM_CALL_SUMMARY_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/util/iterator_range.hpp>

namespace jlm::llvm
{

class CallNode;

/**
 * The CallSummary of a lambda summarizes all call usages of the lambda. It distinguishes between
 * three call usages:
 *
 * 1. The export of the lambda, which is null if the lambda is not exported.
 * 2. All direct calls of the lambda.
 * 3. All other usages, e.g., indirect calls.
 */
class CallSummary final
{
  using DirectCallsConstRange = util::IteratorRange<std::vector<CallNode *>::const_iterator>;
  using OtherUsersConstRange = util::IteratorRange<std::vector<rvsdg::input *>::const_iterator>;

public:
  CallSummary(
      GraphExport * rvsdgExport,
      std::vector<CallNode *> directCalls,
      std::vector<rvsdg::input *> otherUsers)
      : RvsdgExport_(rvsdgExport),
        DirectCalls_(std::move(directCalls)),
        OtherUsers_(std::move(otherUsers))
  {}

  /**
   * Determines whether the lambda is dead.
   *
   * @return True if the lambda is dead, otherwise false.
   */
  [[nodiscard]] bool
  IsDead() const noexcept
  {
    return RvsdgExport_ == nullptr && DirectCalls_.empty() && OtherUsers_.empty();
  }

  /**
   * Determines whether the lambda is exported from the RVSDG
   *
   * @return True if the lambda is exported, otherwise false.
   */
  [[nodiscard]] bool
  IsExported() const noexcept
  {
    return RvsdgExport_ != nullptr;
  }

  /**
   * Determines whether the lambda is only(!) exported from the RVSDG.
   *
   * @return True if the lambda is only exported, otherwise false.
   */
  [[nodiscard]] bool
  IsOnlyExported() const noexcept
  {
    return RvsdgExport_ != nullptr && DirectCalls_.empty() && OtherUsers_.empty();
  }

  /**
   * Determines whether the lambda has only direct calls.
   *
   * @return True if the lambda has only direct calls, otherwise false.
   */
  [[nodiscard]] bool
  HasOnlyDirectCalls() const noexcept
  {
    return RvsdgExport_ == nullptr && OtherUsers_.empty() && !DirectCalls_.empty();
  }

  /**
   * Determines whether the lambda has no other usages, i.e., it can only be exported and/or have
   * direct calls.
   *
   * @return True if the lambda has no other usages, otherwise false.
   */
  [[nodiscard]] bool
  HasNoOtherUsages() const noexcept
  {
    return OtherUsers_.empty();
  }

  /**
   * Determines whether the lambda has only(!) other usages.
   *
   * @return True if the lambda has only other usages, otherwise false.
   */
  [[nodiscard]] bool
  HasOnlyOtherUsages() const noexcept
  {
    return RvsdgExport_ == nullptr && DirectCalls_.empty() && !OtherUsers_.empty();
  }

  /**
   * Returns the number of direct call sites invoking the lambda.
   *
   * @return The number of direct call sites.
   */
  [[nodiscard]] size_t
  NumDirectCalls() const noexcept
  {
    return DirectCalls_.size();
  }

  /**
   * Returns the number of all other users that are not direct calls.
   *
   * @return The number of usages that are not direct calls.
   */
  [[nodiscard]] size_t
  NumOtherUsers() const noexcept
  {
    return OtherUsers_.size();
  }

  /**
   * Returns the export of the lambda.
   *
   * @return The export of the lambda from the RVSDG root region.
   */
  [[nodiscard]] GraphExport *
  GetRvsdgExport() const noexcept
  {
    return RvsdgExport_;
  }

  /**
   * Returns an \ref util::IteratorRange for iterating through all direct call sites.
   *
   * @return An \ref util::IteratorRange of all direct call sites.
   */
  [[nodiscard]] DirectCallsConstRange
  DirectCalls() const noexcept
  {
    return { DirectCalls_.begin(), DirectCalls_.end() };
  }

  /**
   * Returns an \ref util::IteratorRange for iterating through all other usages.
   *
   * @return An \ref util::IteratorRange of all other usages.
   */
  [[nodiscard]] OtherUsersConstRange
  OtherUsers() const noexcept
  {
    return { OtherUsers_.begin(), OtherUsers_.end() };
  }

  /**
   * Creates a new CallSummary.
   *
   * @param rvsdgExport The lambda export.
   * @param directCalls The direct call sites of a lambda.
   * @param otherUsers All other usages of a lambda.
   *
   * @return A new CallSummary instance.
   *
   * @see ComputeCallSummary()
   */
  static std::unique_ptr<CallSummary>
  Create(
      GraphExport * rvsdgExport,
      std::vector<CallNode *> directCalls,
      std::vector<rvsdg::input *> otherUsers)
  {
    return std::make_unique<CallSummary>(
        rvsdgExport,
        std::move(directCalls),
        std::move(otherUsers));
  }

private:
  GraphExport * RvsdgExport_;
  std::vector<CallNode *> DirectCalls_;
  std::vector<rvsdg::input *> OtherUsers_;
};

CallSummary
ComputeCallSummary(const lambda::node & lambdaNode);

}

#endif

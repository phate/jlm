/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_LAMBDA_HPP
#define JLM_LLVM_IR_OPERATORS_LAMBDA_HPP

#include <jlm/llvm/ir/attribute.hpp>
#include <jlm/llvm/ir/linkage.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/util/iterator_range.hpp>

#include <utility>

namespace jlm::llvm
{

class CallNode;

namespace lambda
{

/** \brief Lambda operation
 *
 * A lambda operation determines a lambda's name and \ref FunctionType "function type".
 */
class operation final : public jlm::rvsdg::structural_op
{
public:
  ~operation() override;

  operation(
      std::shared_ptr<const jlm::llvm::FunctionType> type,
      std::string name,
      const jlm::llvm::linkage & linkage,
      jlm::llvm::attributeset attributes)
      : type_(std::move(type)),
        name_(std::move(name)),
        linkage_(linkage),
        attributes_(std::move(attributes))
  {}

  operation(const operation & other) = default;

  operation(operation && other) noexcept = default;

  operation &
  operator=(const operation & other) = default;

  operation &
  operator=(operation && other) noexcept = default;

  [[nodiscard]] const jlm::llvm::FunctionType &
  type() const noexcept
  {
    return *type_;
  }

  [[nodiscard]] const std::shared_ptr<const jlm::llvm::FunctionType> &
  Type() const noexcept
  {
    return type_;
  }

  [[nodiscard]] const std::string &
  name() const noexcept
  {
    return name_;
  }

  [[nodiscard]] const jlm::llvm::linkage &
  linkage() const noexcept
  {
    return linkage_;
  }

  [[nodiscard]] const jlm::llvm::attributeset &
  attributes() const noexcept
  {
    return attributes_;
  }

  [[nodiscard]] std::string
  debug_string() const override;

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override;

  [[nodiscard]] std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

private:
  std::shared_ptr<const jlm::llvm::FunctionType> type_;
  std::string name_;
  jlm::llvm::linkage linkage_;
  jlm::llvm::attributeset attributes_;
};

class cvargument;
class cvinput;
class fctargument;
class output;
class result;

/** \brief Lambda node
 *
 * A lambda node represents a lambda expression in the RVSDG. Its creation requires the invocation
 * of two functions: \ref create() and \ref finalize(). First, a node with only the function
 * arguments is created by invoking \ref create(). The free variables of the lambda expression can
 * then be added to the lambda node using the \ref add_ctxvar() method, and the body of the lambda
 * node can be created. Finally, the lambda node can be finalized by invoking \ref finalize().
 *
 * The following snippet illustrates the creation of lambda nodes:
 *
 * \code{.cpp}
 *   auto lambda = lambda::node::create(...);
 *   ...
 *   auto cv1 = lambda->add_ctxvar(...);
 *   auto cv2 = lambda->add_ctxvar(...);
 *   ...
 *   // generate lambda body
 *   ...
 *   auto output = lambda->finalize(...);
 * \endcode
 */
class node final : public rvsdg::StructuralNode
{
public:
  class CallSummary;

private:
  class cviterator;
  class cvconstiterator;

  class fctargiterator;
  class fctargconstiterator;

  class fctresiterator;
  class fctresconstiterator;

  using fctargument_range = jlm::util::iterator_range<fctargiterator>;
  using fctargument_constrange = jlm::util::iterator_range<fctargconstiterator>;

  using ctxvar_range = jlm::util::iterator_range<cviterator>;
  using ctxvar_constrange = jlm::util::iterator_range<cvconstiterator>;

  using fctresult_range = jlm::util::iterator_range<fctresiterator>;
  using fctresult_constrange = jlm::util::iterator_range<fctresconstiterator>;

public:
  ~node() override;

private:
  node(rvsdg::Region * parent, lambda::operation && op)
      : StructuralNode(op, parent, 1)
  {}

public:
  [[nodiscard]] fctargument_range
  fctarguments();

  [[nodiscard]] fctargument_constrange
  fctarguments() const;

  ctxvar_range
  ctxvars();

  [[nodiscard]] ctxvar_constrange
  ctxvars() const;

  fctresult_range
  fctresults();

  [[nodiscard]] fctresult_constrange
  fctresults() const;

  [[nodiscard]] rvsdg::Region *
  subregion() const noexcept
  {
    return StructuralNode::subregion(0);
  }

  [[nodiscard]] const lambda::operation &
  operation() const noexcept
  {
    return *jlm::util::AssertedCast<const lambda::operation>(&StructuralNode::operation());
  }

  [[nodiscard]] const jlm::llvm::FunctionType &
  type() const noexcept
  {
    return operation().type();
  }

  [[nodiscard]] const std::shared_ptr<const jlm::llvm::FunctionType> &
  Type() const noexcept
  {
    return operation().Type();
  }

  [[nodiscard]] const std::string &
  name() const noexcept
  {
    return operation().name();
  }

  [[nodiscard]] const jlm::llvm::linkage &
  linkage() const noexcept
  {
    return operation().linkage();
  }

  [[nodiscard]] const jlm::llvm::attributeset &
  attributes() const noexcept
  {
    return operation().attributes();
  }

  [[nodiscard]] size_t
  ncvarguments() const noexcept
  {
    return ninputs();
  }

  [[nodiscard]] size_t
  nfctarguments() const noexcept
  {
    return subregion()->narguments() - ninputs();
  }

  [[nodiscard]] size_t
  nfctresults() const noexcept
  {
    return subregion()->nresults();
  }

  /**
   * Adds a context/free variable to the lambda node. The \p origin must be from the same region
   * as the lambda node.
   *
   * \return The context variable argument from the lambda region.
   */
  lambda::cvargument *
  add_ctxvar(jlm::rvsdg::output * origin);

  /**
   * Remove lambda inputs and their respective arguments.
   *
   * An input must match the condition specified by \p match and its argument must be dead.
   *
   * @tparam F A type that supports the function call operator: bool operator(const cvinput&)
   * @param match Defines the condition of the elements to remove.
   * @return The number of removed inputs.
   *
   * \see cvargument#IsDead()
   */
  template<typename F>
  size_t
  RemoveLambdaInputsWhere(const F & match);

  /**
   * Remove all dead inputs.
   *
   * @return The number of removed inputs.
   *
   * \see RemoveLambdaInputsWhere()
   */
  size_t
  PruneLambdaInputs()
  {
    auto match = [](const cvinput &)
    {
      return true;
    };

    return RemoveLambdaInputsWhere(match);
  }

  [[nodiscard]] cvinput *
  input(size_t n) const noexcept;

  [[nodiscard]] lambda::output *
  output() const noexcept;

  [[nodiscard]] lambda::fctargument *
  fctargument(size_t n) const noexcept;

  [[nodiscard]] lambda::cvargument *
  cvargument(size_t n) const noexcept;

  [[nodiscard]] lambda::result *
  fctresult(size_t n) const noexcept;

  lambda::node *
  copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::output *> & operands) const override;

  lambda::node *
  copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const override;

  /**
   * @return The memory state argument of the lambda subregion.
   */
  [[nodiscard]] rvsdg::RegionArgument &
  GetMemoryStateRegionArgument() const noexcept;

  /**
   * @return The memory state result of the lambda subregion.
   */
  [[nodiscard]] rvsdg::RegionResult &
  GetMemoryStateRegionResult() const noexcept;

  /**
   *
   * @param lambdaNode The lambda node for which to retrieve the
   * LambdaEntryMemoryStateSplitOperation node.
   * @return The LambdaEntryMemoryStateSplitOperation node connected to the memory state input if
   * present, otherwise nullptr.
   *
   * @see GetMemoryStateExitMerge()
   */
  static rvsdg::simple_node *
  GetMemoryStateEntrySplit(const lambda::node & lambdaNode) noexcept;

  /**
   *
   * @param lambdaNode The lambda node for which to retrieve the
   * LambdaExitMemoryStateMergeOperation node.
   * @return The LambdaExitMemoryStateMergeOperation node connected to the memory state output if
   * present, otherwise nullptr.
   *
   * @see GetMemoryStateEntrySplit()
   */
  [[nodiscard]] static rvsdg::simple_node *
  GetMemoryStateExitMerge(const lambda::node & lambdaNode) noexcept;

  /**
   * Creates a lambda node in the region \p parent with the function type \p type and name \p name.
   * After the invocation of \ref create(), the lambda node only features the function arguments.
   * Free variables can be added to the function node using \ref add_ctxvar(). The generation of the
   * node can be finished using the \ref finalize() method.
   *
   * \param parent The region where the lambda node is created.
   * \param type The lambda node's type.
   * \param name The lambda node's name.
   * \param linkage The lambda node's linkage.
   * \param attributes The lambda node's attributes.
   *
   * \return A lambda node featuring only function arguments.
   */
  static node *
  create(
      rvsdg::Region * parent,
      std::shared_ptr<const jlm::llvm::FunctionType> type,
      const std::string & name,
      const jlm::llvm::linkage & linkage,
      const jlm::llvm::attributeset & attributes);

  /**
   * See \ref create().
   */
  static node *
  create(
      rvsdg::Region * parent,
      std::shared_ptr<const jlm::llvm::FunctionType> type,
      const std::string & name,
      const jlm::llvm::linkage & linkage)
  {
    return create(parent, type, name, linkage, {});
  }

  /**
   * Finalizes the creation of a lambda node.
   *
   * \param results The result values of the lambda expression, originating from the lambda region.
   *
   * \return The output of the lambda node.
   */
  lambda::output *
  finalize(const std::vector<jlm::rvsdg::output *> & results);

  /**
   * Compute the \ref CallSummary of the lambda.
   *
   * @return A new CallSummary instance.
   */
  [[nodiscard]] std::unique_ptr<CallSummary>
  ComputeCallSummary() const;

  /**
   * Determines whether \p lambdaNode is exported from the module.
   *
   * @param lambdaNode The lambda node to be checked.
   * @return True if \p lambdaNode is exported, otherwise false.
   *
   * \note This method is equivalent to invoking CallSummary::IsExported().
   */
  [[nodiscard]] static bool
  IsExported(const lambda::node & lambdaNode);
};

/** \brief Lambda context variable input
 */
class cvinput final : public jlm::rvsdg::structural_input
{
  friend ::jlm::llvm::lambda::node;

public:
  ~cvinput() override;

private:
  cvinput(lambda::node * node, jlm::rvsdg::output * origin)
      : structural_input(node, origin, origin->Type())
  {}

  static cvinput *
  create(lambda::node * node, jlm::rvsdg::output * origin)
  {
    auto input = std::unique_ptr<cvinput>(new cvinput(node, origin));
    return jlm::util::AssertedCast<cvinput>(node->append_input(std::move(input)));
  }

public:
  [[nodiscard]] cvargument *
  argument() const noexcept;

  [[nodiscard]] lambda::node *
  node() const noexcept
  {
    return jlm::util::AssertedCast<lambda::node>(structural_input::node());
  }
};

/** \brief Lambda context variable iterator
 */
class node::cviterator final : public jlm::rvsdg::input::iterator<cvinput>
{
  friend ::jlm::llvm::lambda::node;

  constexpr explicit cviterator(cvinput * input)
      : jlm::rvsdg::input::iterator<cvinput>(input)
  {}

  [[nodiscard]] cvinput *
  next() const override
  {
    auto node = value()->node();
    auto index = value()->index();

    return node->ninputs() > index + 1 ? node->input(index + 1) : nullptr;
  }
};

/** \brief Lambda context variable const iterator
 */
class node::cvconstiterator final : public jlm::rvsdg::input::constiterator<cvinput>
{
  friend ::jlm::llvm::lambda::node;

  constexpr explicit cvconstiterator(const cvinput * input)
      : jlm::rvsdg::input::constiterator<cvinput>(input)
  {}

  [[nodiscard]] const cvinput *
  next() const override
  {
    auto node = value()->node();
    auto index = value()->index();

    return node->ninputs() > index + 1 ? node->input(index + 1) : nullptr;
  }
};

/** \brief Lambda output
 */
class output final : public rvsdg::StructuralOutput
{
  friend ::jlm::llvm::lambda::node;

public:
  ~output() override;

  output(lambda::node * node, std::shared_ptr<const rvsdg::Type> type)
      : StructuralOutput(node, std::move(type))
  {}

private:
  static output *
  create(lambda::node * node, std::shared_ptr<const rvsdg::Type> type)
  {
    auto output = std::make_unique<lambda::output>(node, std::move(type));
    return jlm::util::AssertedCast<lambda::output>(node->append_output(std::move(output)));
  }

public:
  lambda::node *
  node() const noexcept
  {
    return jlm::util::AssertedCast<lambda::node>(StructuralOutput::node());
  }
};

/** \brief Lambda function argument
 */
class fctargument final : public rvsdg::RegionArgument
{
  friend ::jlm::llvm::lambda::node;

public:
  ~fctargument() override;

  const jlm::llvm::attributeset &
  attributes() const noexcept
  {
    return attributes_;
  }

  void
  set_attributes(const jlm::llvm::attributeset & attributes)
  {
    attributes_ = attributes;
  }

  fctargument &
  Copy(rvsdg::Region & region, rvsdg::structural_input * input) override;

private:
  fctargument(rvsdg::Region * region, std::shared_ptr<const jlm::rvsdg::Type> type)
      : rvsdg::RegionArgument(region, nullptr, std::move(type))
  {}

  static fctargument *
  create(rvsdg::Region * region, std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    auto argument = new fctargument(region, std::move(type));
    region->append_argument(argument);
    return argument;
  }

  jlm::llvm::attributeset attributes_;
};

/** \brief Lambda function argument iterator
 */
class node::fctargiterator final : public jlm::rvsdg::output::iterator<lambda::fctargument>
{
  friend ::jlm::llvm::lambda::node;

  constexpr explicit fctargiterator(lambda::fctargument * argument)
      : jlm::rvsdg::output::iterator<lambda::fctargument>(argument)
  {}

  [[nodiscard]] lambda::fctargument *
  next() const override
  {
    auto index = value()->index();
    auto lambda = jlm::util::AssertedCast<lambda::node>(value()->region()->node());

    /*
      This assumes that all function arguments were added to the lambda region
      before any context variable was added.
    */
    return lambda->nfctarguments() > index + 1 ? lambda->fctargument(index + 1) : nullptr;
  }
};

/** \brief Lambda function argument const iterator
 */
class node::fctargconstiterator final
    : public jlm::rvsdg::output::constiterator<lambda::fctargument>
{
  friend ::jlm::llvm::lambda::node;

  constexpr explicit fctargconstiterator(const lambda::fctargument * argument)
      : jlm::rvsdg::output::constiterator<lambda::fctargument>(argument)
  {}

  [[nodiscard]] const lambda::fctargument *
  next() const override
  {
    auto index = value()->index();
    auto lambda = jlm::util::AssertedCast<lambda::node>(value()->region()->node());

    /*
      This assumes that all function arguments were added to the lambda region
      before any context variable was added.
    */
    return lambda->nfctarguments() > index + 1 ? lambda->fctargument(index + 1) : nullptr;
  }
};

/** \brief Lambda context variable argument
 */
class cvargument final : public rvsdg::RegionArgument
{
  friend ::jlm::llvm::lambda::node;

public:
  ~cvargument() override;

  cvargument &
  Copy(rvsdg::Region & region, jlm::rvsdg::structural_input * input) override;

private:
  cvargument(rvsdg::Region * region, cvinput * input)
      : rvsdg::RegionArgument(region, input, input->Type())
  {}

  static cvargument *
  create(rvsdg::Region * region, lambda::cvinput * input)
  {
    auto argument = new cvargument(region, input);
    region->append_argument(argument);
    return argument;
  }

public:
  cvinput *
  input() const noexcept
  {
    return jlm::util::AssertedCast<cvinput>(rvsdg::RegionArgument::input());
  }
};

/** \brief Lambda result
 */
class result final : public rvsdg::RegionResult
{
  friend ::jlm::llvm::lambda::node;

public:
  ~result() override;

  result &
  Copy(rvsdg::output & origin, rvsdg::StructuralOutput * output) override;

private:
  explicit result(jlm::rvsdg::output * origin)
      : rvsdg::RegionResult(origin->region(), origin, nullptr, origin->Type())
  {}

  static result *
  create(jlm::rvsdg::output * origin)
  {
    auto result = new lambda::result(origin);
    origin->region()->append_result(result);
    return result;
  }

public:
  lambda::output *
  output() const noexcept
  {
    return jlm::util::AssertedCast<lambda::output>(rvsdg::RegionResult::output());
  }
};

/** \brief Lambda result iterator
 */
class node::fctresiterator final : public jlm::rvsdg::input::iterator<lambda::result>
{
  friend ::jlm::llvm::lambda::node;

  constexpr explicit fctresiterator(lambda::result * result)
      : jlm::rvsdg::input::iterator<lambda::result>(result)
  {}

  [[nodiscard]] lambda::result *
  next() const override
  {
    auto index = value()->index();
    auto lambda = jlm::util::AssertedCast<lambda::node>(value()->region()->node());

    return lambda->nfctresults() > index + 1 ? lambda->fctresult(index + 1) : nullptr;
  }
};

/** \brief Lambda result const iterator
 */
class node::fctresconstiterator final : public jlm::rvsdg::input::constiterator<lambda::result>
{
  friend ::jlm::llvm::lambda::node;

  constexpr explicit fctresconstiterator(const lambda::result * result)
      : jlm::rvsdg::input::constiterator<lambda::result>(result)
  {}

  [[nodiscard]] const lambda::result *
  next() const override
  {
    auto index = value()->index();
    auto lambda = jlm::util::AssertedCast<lambda::node>(value()->region()->node());

    return lambda->nfctresults() > index + 1 ? lambda->fctresult(index + 1) : nullptr;
  }
};

/**
 * The CallSummary of a lambda summarizes all call usages of the lambda. It distinguishes between
 * three call usages:
 *
 * 1. The export of the lambda, which is null if the lambda is not exported.
 * 2. All direct calls of the lambda.
 * 3. All other usages, e.g., indirect calls.
 */
class node::CallSummary final
{
  using DirectCallsConstRange = util::iterator_range<std::vector<CallNode *>::const_iterator>;
  using OtherUsersConstRange = util::iterator_range<std::vector<rvsdg::input *>::const_iterator>;

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
   * Returns an \ref util::iterator_range for iterating through all direct call sites.
   *
   * @return An \ref util::iterator_range of all direct call sites.
   */
  [[nodiscard]] DirectCallsConstRange
  DirectCalls() const noexcept
  {
    return { DirectCalls_.begin(), DirectCalls_.end() };
  }

  /**
   * Returns an \ref util::iterator_range for iterating through all other usages.
   *
   * @return An \ref util::iterator_range of all other usages.
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

template<typename F>
size_t
lambda::node::RemoveLambdaInputsWhere(const F & match)
{
  size_t numRemovedInputs = 0;

  // iterate backwards to avoid the invalidation of 'n' by RemoveInput()
  for (size_t n = ninputs() - 1; n != static_cast<size_t>(-1); n--)
  {
    auto & lambdaInput = *input(n);
    auto & argument = *lambdaInput.argument();

    if (argument.IsDead() && match(lambdaInput))
    {
      subregion()->RemoveArgument(argument.index());
      RemoveInput(n);
      numRemovedInputs++;
    }
  }

  return numRemovedInputs;
}

}
}

#endif

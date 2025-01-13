/*
 * Copyright 2012 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_PHI_HPP
#define JLM_LLVM_IR_OPERATORS_PHI_HPP

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/util/common.hpp>

namespace jlm::llvm
{

namespace lambda
{
class node;
}

namespace phi
{

/* phi operation class  */

class operation final : public rvsdg::StructuralOperation
{
public:
  ~operation() override;

  virtual std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;
};

/* phi node class */

class builder;
class cvargument;
class cvinput;
class rvoutput;

class node final : public rvsdg::StructuralNode
{
  friend class phi::builder;

  class cvconstiterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const cvinput *;
    using difference_type = std::ptrdiff_t;
    using pointer = const cvinput **;
    using reference = const cvinput *&;

  private:
    friend class phi::node;

    cvconstiterator(const cvinput * input)
        : input_(input)
    {}

  public:
    const cvinput *
    input() const noexcept
    {
      return input_;
    }

    const cvinput &
    operator*() const
    {
      JLM_ASSERT(input_ != nullptr);
      return *input_;
    }

    const cvinput *
    operator->() const
    {
      return input_;
    }

    cvconstiterator &
    operator++();

    cvconstiterator
    operator++(int)
    {
      cvconstiterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool
    operator==(const cvconstiterator & other) const
    {
      return input_ == other.input_;
    }

    bool
    operator!=(const cvconstiterator & other) const
    {
      return !operator==(other);
    }

  private:
    const cvinput * input_;
  };

  class cviterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = cvinput *;
    using difference_type = std::ptrdiff_t;
    using pointer = cvinput **;
    using reference = cvinput *&;

  private:
    friend class phi::node;

    cviterator(cvinput * input)
        : input_(input)
    {}

  public:
    cvinput *
    input() const noexcept
    {
      return input_;
    }

    cvinput &
    operator*() const
    {
      JLM_ASSERT(input_ != nullptr);
      return *input_;
    }

    cvinput *
    operator->() const
    {
      return input_;
    }

    cviterator &
    operator++();

    cviterator
    operator++(int)
    {
      cviterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool
    operator==(const cviterator & other) const
    {
      return input_ == other.input_;
    }

    bool
    operator!=(const cviterator & other) const
    {
      return !operator==(other);
    }

  private:
    cvinput * input_;
  };

  class rvconstiterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const rvoutput *;
    using difference_type = std::ptrdiff_t;
    using pointer = const rvoutput **;
    using reference = const rvoutput *&;

    friend class phi::node;

    rvconstiterator(const rvoutput * output)
        : output_(output)
    {}

  public:
    const rvoutput *
    output() const noexcept
    {
      return output_;
    }

    const rvoutput &
    operator*() const
    {
      JLM_ASSERT(output_ != nullptr);
      return *output_;
    }

    const rvoutput *
    operator->() const
    {
      return output_;
    }

    rvconstiterator &
    operator++();

    rvconstiterator
    operator++(int)
    {
      rvconstiterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool
    operator==(const rvconstiterator & other) const
    {
      return output_ == other.output_;
    }

    bool
    operator!=(const rvconstiterator & other) const
    {
      return !operator==(other);
    }

  private:
    const rvoutput * output_;
  };

  class rviterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = rvoutput *;
    using difference_type = std::ptrdiff_t;
    using pointer = rvoutput **;
    using reference = rvoutput *&;

  private:
    friend class phi::node;

    rviterator(rvoutput * output)
        : output_(output)
    {}

  public:
    rvoutput *
    output() const noexcept
    {
      return output_;
    }

    rvoutput &
    operator*() const
    {
      JLM_ASSERT(output_ != nullptr);
      return *output_;
    }

    rvoutput *
    operator->() const
    {
      return output_;
    }

    rviterator &
    operator++();

    rviterator
    operator++(int)
    {
      rviterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool
    operator==(const rviterator & other) const
    {
      return output_ == other.output_;
    }

    bool
    operator!=(const rviterator & other) const
    {
      return !operator==(other);
    }

  private:
    rvoutput * output_;
  };

public:
  ~node() override;

private:
  explicit node(rvsdg::Region * parent)
      : StructuralNode(parent, 1)
  {}

  static phi::node *
  create(rvsdg::Region * parent)
  {
    return new phi::node(parent);
  }

public:
  [[nodiscard]] const phi::operation &
  GetOperation() const noexcept override;

  cvconstiterator
  begin_cv() const
  {
    if (ninputs() == 0)
      return end_cv();

    return cvconstiterator(input(0));
  }

  cviterator
  begin_cv()
  {
    if (ninputs() == 0)
      return end_cv();

    return cviterator(input(0));
  }

  cvconstiterator
  end_cv() const
  {
    return cvconstiterator(nullptr);
  }

  cviterator
  end_cv()
  {
    return cviterator(nullptr);
  }

  rvconstiterator
  begin_rv() const
  {
    if (noutputs() == 0)
      return end_rv();

    return rvconstiterator(output(0));
  }

  rviterator
  begin_rv()
  {
    if (noutputs() == 0)
      return end_rv();

    return rviterator(output(0));
  }

  rvconstiterator
  end_rv() const
  {
    return rvconstiterator(nullptr);
  }

  rviterator
  end_rv()
  {
    return rviterator(nullptr);
  }

  rvsdg::Region *
  subregion() const noexcept
  {
    return StructuralNode::subregion(0);
  }

  cvargument *
  add_ctxvar(jlm::rvsdg::output * origin);

  /**
   * Remove phi arguments and their respective inputs.
   *
   * An argument must match the condition specified by \p match and it must be dead.
   *
   * @tparam F A type that supports the function call operator: bool operator(const argument&)
   * @param match Defines the condition of the elements to remove.
   * @return The number of removed arguments.
   *
   * \note The application of this method might leave the phi node in an invalid state. Some
   * outputs might refer to arguments that have been removed by the application of this method. It
   * is up to the caller to ensure that the invariants of the phi node will eventually be met again.
   *
   * \see RegionArgument#IsDead()
   * \see PrunePhiArguments()
   * \see RemovePhiOutputsWhere()
   * \see PrunePhiOutputs()
   */
  template<typename F>
  size_t
  RemovePhiArgumentsWhere(const F & match);

  /**
   * Remove all dead phi arguments and their respective inputs.
   *
   * @return The number of removed arguments.
   *
   * \note The application of this method might leave the phi node in an invalid state. Some
   * outputs might refer to arguments that have been removed by the application of this method. It
   * is up to the caller to ensure that the invariants of the phi node will eventually be met again.
   *
   * \see RemovePhiArgumentsWhere()
   */
  size_t
  PrunePhiArguments()
  {
    auto match = [](const rvsdg::RegionArgument &)
    {
      return true;
    };

    return RemovePhiArgumentsWhere(match);
  }

  /**
   * Remove phi outputs and their respective results.
   *
   * An output must match the condition specified by \p match and it must be dead.
   *
   * @tparam F A type that supports the function call operator: bool operator(const phi::rvoutput&)
   * @param match Defines the condition of the elements to remove.
   * @return The number of removed outputs.
   *
   * \note The application of this method might leave the phi node in an invalid state. Some
   * arguments might refer to outputs that have been removed by the application of this method. It
   * is up to the caller to ensure that the invariants of the phi node will eventually be met again.
   *
   * \see rvoutput#IsDead()
   * \see PrunePhiOutputs()
   * \see RemovePhiArgumentsWhere()
   * \see PrunePhiArguments()
   */
  template<typename F>
  size_t
  RemovePhiOutputsWhere(const F & match);

  /**
   * Remove all dead phi outputs and their respective results.
   *
   * @return The number of removed outputs.
   *
   * \note The application of this method might leave the phi node in an invalid state. Some
   * arguments might refer to outputs that have been removed by the application of this method. It
   * is up to the caller to ensure that the invariants of the phi node will eventually be met again.
   *
   * \see RemovePhiOutputsWhere()
   * \see rvoutput#IsDead()
   */
  size_t
  PrunePhiOutputs()
  {
    auto match = [](const phi::rvoutput &)
    {
      return true;
    };

    return RemovePhiOutputsWhere(match);
  }

  cvinput *
  input(size_t n) const noexcept;

  rvoutput *
  output(size_t n) const noexcept;

  virtual phi::node *
  copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const override;

  /**
   * Extracts all lambda nodes from a phi node.
   *
   * The function is capable of handling nested phi nodes.
   *
   * @param phiNode The phi node from which the lambda nodes should be extracted.
   * @return A vector of lambda nodes.
   */
  static std::vector<lambda::node *>
  ExtractLambdaNodes(const phi::node & phiNode);
};

/* phi builder class */

class rvoutput;

class builder final
{
public:
  constexpr builder() noexcept
      : node_(nullptr)
  {}

  rvsdg::Region *
  subregion() const noexcept
  {
    return node_ ? node_->subregion() : nullptr;
  }

  void
  begin(rvsdg::Region * parent)
  {
    if (node_)
      return;

    node_ = phi::node::create(parent);
  }

  phi::cvargument *
  add_ctxvar(jlm::rvsdg::output * origin)
  {
    if (!node_)
      return nullptr;

    return node_->add_ctxvar(origin);
  }

  phi::rvoutput *
  add_recvar(std::shared_ptr<const jlm::rvsdg::Type> type);

  phi::node *
  end();

private:
  phi::node * node_;
};

/* phi context variable input class */

class cvinput final : public rvsdg::StructuralInput
{
  friend class phi::node;

public:
  ~cvinput() override;

  cvinput(
      phi::node * node,
      jlm::rvsdg::output * origin,
      std::shared_ptr<const jlm::rvsdg::Type> type)
      : StructuralInput(node, origin, std::move(type))
  {}

private:
  cvinput(const cvinput &) = delete;

  cvinput(cvinput &&) = delete;

  cvinput &
  operator=(const cvinput &) = delete;

  cvinput &
  operator=(cvinput &&) = delete;

  static cvinput *
  create(
      phi::node * node,
      jlm::rvsdg::output * origin,
      std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    auto input = std::make_unique<cvinput>(node, origin, std::move(type));
    return static_cast<cvinput *>(node->append_input(std::move(input)));
  }

public:
  cvargument *
  argument() const noexcept;

  phi::node *
  node() const noexcept
  {
    return static_cast<phi::node *>(StructuralInput::node());
  }
};

/* phi recursion variable output class */

class rvargument;
class rvresult;

class rvoutput final : public rvsdg::StructuralOutput
{
  friend class phi::builder;

public:
  ~rvoutput() override;

private:
  rvoutput(phi::node * node, rvargument * argument, std::shared_ptr<const rvsdg::Type> type)
      : StructuralOutput(node, std::move(type)),
        argument_(argument)
  {}

  rvoutput(const rvoutput &) = delete;

  rvoutput(rvoutput &&) = delete;

  rvoutput &
  operator=(const rvoutput &) = delete;

  rvoutput &
  operator=(rvoutput &&) = delete;

  static rvoutput *
  create(phi::node * node, rvargument * argument, std::shared_ptr<const rvsdg::Type> type);

public:
  rvargument *
  argument() const noexcept
  {
    return argument_;
  }

  rvresult *
  result() const noexcept;

  void
  set_rvorigin(jlm::rvsdg::output * origin);

  phi::node *
  node() const noexcept
  {
    return static_cast<phi::node *>(StructuralOutput::node());
  }

private:
  rvargument * argument_;
};

/* phi recursion variable argument class */

class rvresult;

class rvargument final : public rvsdg::RegionArgument
{
  friend class phi::builder;
  friend class phi::rvoutput;

public:
  ~rvargument() override;

private:
  rvargument(rvsdg::Region * region, const std::shared_ptr<const jlm::rvsdg::Type> type)
      : RegionArgument(region, nullptr, std::move(type)),
        output_(nullptr)
  {}

  rvargument(const rvargument &) = delete;

  rvargument(rvargument &&) = delete;

  rvargument &
  operator=(const rvargument &) = delete;

  rvargument &
  operator=(rvargument &&) = delete;

  static rvargument *
  create(rvsdg::Region * region, std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    auto argument = new rvargument(region, std::move(type));
    region->append_argument(argument);
    return argument;
  }

public:
  rvoutput *
  output() const noexcept
  {
    JLM_ASSERT(output_ != nullptr);
    return output_;
  }

  rvresult *
  result() const noexcept
  {
    return output()->result();
  }

  rvargument &
  Copy(rvsdg::Region & region, rvsdg::StructuralInput * input) override;

private:
  rvoutput * output_;
};

/* phi context variable argument class */

class cvinput;
class node;

class cvargument final : public rvsdg::RegionArgument
{
  friend class phi::node;

public:
  ~cvargument() override;

  cvargument(rvsdg::Region * region, phi::cvinput * input, std::shared_ptr<const rvsdg::Type> type)
      : rvsdg::RegionArgument(region, input, std::move(type))
  {}

private:
  cvargument(const cvargument &) = delete;

  cvargument(cvargument &&) = delete;

  cvargument &
  operator=(const cvargument &) = delete;

  cvargument &
  operator=(cvargument &&) = delete;

  cvargument &
  Copy(rvsdg::Region & region, rvsdg::StructuralInput * input) override;

  static cvargument *
  create(rvsdg::Region * region, phi::cvinput * input, std::shared_ptr<const rvsdg::Type> type)
  {
    auto argument = new cvargument(region, input, std::move(type));
    region->append_argument(argument);
    return argument;
  }

public:
  cvinput *
  input() const noexcept
  {
    return static_cast<cvinput *>(RegionArgument::input());
  }
};

/* phi recursion variable result class */

class rvresult final : public rvsdg::RegionResult
{
  friend class phi::builder;

public:
  ~rvresult() override;

private:
  rvresult(
      rvsdg::Region * region,
      jlm::rvsdg::output * origin,
      rvoutput * output,
      std::shared_ptr<const rvsdg::Type> type)
      : RegionResult(region, origin, output, std::move(type))
  {}

  rvresult(const rvresult &) = delete;

  rvresult(rvresult &&) = delete;

  rvresult &
  operator=(const rvresult &) = delete;

  rvresult &
  operator=(rvresult &&) = delete;

  rvresult &
  Copy(rvsdg::output & origin, rvsdg::StructuralOutput * output) override;

  static rvresult *
  create(
      rvsdg::Region * region,
      jlm::rvsdg::output * origin,
      rvoutput * output,
      std::shared_ptr<const rvsdg::Type> type)
  {
    auto result = new rvresult(region, origin, output, type);
    region->append_result(result);
    return result;
  }

public:
  rvoutput *
  output() const noexcept
  {
    return static_cast<rvoutput *>(RegionResult::output());
  }

  rvargument *
  argument() const noexcept
  {
    return output()->argument();
  }
};

/* method definitions */

inline node::cvconstiterator &
node::cvconstiterator::operator++()
{
  if (input_ == nullptr)
    return *this;

  auto node = input_->node();
  auto index = input_->index();
  JLM_ASSERT(node->ninputs() != 0);
  input_ = node->ninputs() - 1 == index ? nullptr : node->input(index + 1);

  return *this;
}

inline node::cviterator &
node::cviterator::operator++()
{
  if (input_ == nullptr)
    return *this;

  auto node = input_->node();
  auto index = input_->index();
  JLM_ASSERT(node->ninputs() != 0);
  input_ = node->ninputs() - 1 == index ? nullptr : node->input(index + 1);

  return *this;
}

inline node::rvconstiterator &
node::rvconstiterator::operator++()
{
  if (output_ == nullptr)
    return *this;

  auto index = output_->index();
  auto node = output_->node();
  JLM_ASSERT(node->noutputs() != 0);
  output_ = node->noutputs() - 1 == index ? nullptr : node->output(index + 1);

  return *this;
}

inline node::rviterator &
node::rviterator::operator++()
{
  if (output_ == nullptr)
    return *this;

  auto index = output_->index();
  auto node = output_->node();
  JLM_ASSERT(node->noutputs() != 0);
  output_ = node->noutputs() - 1 == index ? nullptr : node->output(index + 1);

  return *this;
}

inline cvargument *
cvinput::argument() const noexcept
{
  JLM_ASSERT(arguments.size() == 1);
  return static_cast<cvargument *>(arguments.first());
}

inline rvoutput *
rvoutput::create(phi::node * node, rvargument * argument, std::shared_ptr<const rvsdg::Type> type)
{
  JLM_ASSERT(argument->type() == *type);
  auto output = std::unique_ptr<rvoutput>(new rvoutput(node, argument, std::move(type)));
  return static_cast<rvoutput *>(node->append_output(std::move(output)));
}

inline rvresult *
rvoutput::result() const noexcept
{
  JLM_ASSERT(results.size() == 1);
  return static_cast<rvresult *>(results.first());
}

inline void
rvoutput::set_rvorigin(jlm::rvsdg::output * origin)
{
  JLM_ASSERT(result()->origin() == argument());
  result()->divert_to(origin);
}

template<typename F>
size_t
phi::node::RemovePhiArgumentsWhere(const F & match)
{
  size_t numRemovedArguments = 0;

  // iterate backwards to avoid the invalidation of 'n' by RemoveArgument()
  for (size_t n = subregion()->narguments() - 1; n != static_cast<size_t>(-1); n--)
  {
    auto & argument = *subregion()->argument(n);
    auto input = argument.input();

    if (argument.IsDead() && match(argument))
    {
      subregion()->RemoveArgument(argument.index());
      numRemovedArguments++;

      if (input)
      {
        RemoveInput(input->index());
      }
    }
  }

  return numRemovedArguments;
}

template<typename F>
size_t
phi::node::RemovePhiOutputsWhere(const F & match)
{
  size_t numRemovedOutputs = 0;

  // iterate backwards to avoid the invalidation of 'n' by RemoveOutput()
  for (size_t n = noutputs() - 1; n != static_cast<size_t>(-1); n--)
  {
    auto & phiOutput = *output(n);
    auto & phiResult = *phiOutput.result();

    if (phiOutput.IsDead() && match(phiOutput))
    {
      subregion()->RemoveResult(phiResult.index());
      RemoveOutput(phiOutput.index());
      numRemovedOutputs++;
    }
  }

  return numRemovedOutputs;
}

}

}

#endif

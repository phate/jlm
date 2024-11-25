/*
 * Copyright 2010 2011 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_GAMMA_HPP
#define JLM_RVSDG_GAMMA_HPP

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/structural-normal-form.hpp>

namespace jlm::rvsdg
{

/* gamma normal form */

class gamma_normal_form final : public structural_normal_form
{
public:
  virtual ~gamma_normal_form() noexcept;

  gamma_normal_form(
      const std::type_info & operator_class,
      jlm::rvsdg::node_normal_form * parent,
      Graph * graph) noexcept;

  virtual bool
  normalize_node(jlm::rvsdg::node * node) const override;

  virtual void
  set_predicate_reduction(bool enable);

  inline bool
  get_predicate_reduction() const noexcept
  {
    return enable_predicate_reduction_;
  }

  virtual void
  set_invariant_reduction(bool enable);

  inline bool
  get_invariant_reduction() const noexcept
  {
    return enable_invariant_reduction_;
  }

  virtual void
  set_control_constant_reduction(bool enable);

  inline bool
  get_control_constant_reduction() const noexcept
  {
    return enable_control_constant_reduction_;
  }

private:
  bool enable_predicate_reduction_;
  bool enable_invariant_reduction_;
  bool enable_control_constant_reduction_;
};

/* gamma operation */

class output;
class Type;

class GammaOperation final : public structural_op
{
public:
  ~GammaOperation() noexcept override;

  explicit constexpr GammaOperation(size_t nalternatives) noexcept
      : structural_op(),
        nalternatives_(nalternatives)
  {}

  inline size_t
  nalternatives() const noexcept
  {
    return nalternatives_;
  }

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  virtual bool
  operator==(const operation & other) const noexcept override;

  static jlm::rvsdg::gamma_normal_form *
  normal_form(Graph * graph) noexcept
  {
    return static_cast<jlm::rvsdg::gamma_normal_form *>(
        graph->node_normal_form(typeid(GammaOperation)));
  }

private:
  size_t nalternatives_;
};

/* gamma node */

class GammaInput;
class GammaOutput;

class GammaNode : public StructuralNode
{
public:
  ~GammaNode() noexcept override;

private:
  GammaNode(rvsdg::output * predicate, size_t nalternatives);

  class entryvar_iterator
  {
  public:
    constexpr entryvar_iterator(GammaInput * input) noexcept
        : input_(input)
    {}

    GammaInput *
    input() const noexcept
    {
      return input_;
    }

    const entryvar_iterator &
    operator++() noexcept;

    inline const entryvar_iterator
    operator++(int) noexcept
    {
      entryvar_iterator it(*this);
      ++(*this);
      return it;
    }

    inline bool
    operator==(const entryvar_iterator & other) const noexcept
    {
      return input_ == other.input_;
    }

    inline bool
    operator!=(const entryvar_iterator & other) const noexcept
    {
      return !(*this == other);
    }

    GammaInput &
    operator*() noexcept
    {
      return *input_;
    }

    GammaInput *
    operator->() noexcept
    {
      return input_;
    }

  private:
    GammaInput * input_;
  };

  class exitvar_iterator
  {
  public:
    constexpr explicit exitvar_iterator(GammaOutput * output) noexcept
        : output_(output)
    {}

    [[nodiscard]] GammaOutput *
    output() const noexcept
    {
      return output_;
    }

    const exitvar_iterator &
    operator++() noexcept;

    inline const exitvar_iterator
    operator++(int) noexcept
    {
      exitvar_iterator it(*this);
      ++(*this);
      return it;
    }

    inline bool
    operator==(const exitvar_iterator & other) const noexcept
    {
      return output_ == other.output_;
    }

    inline bool
    operator!=(const exitvar_iterator & other) const noexcept
    {
      return !(*this == other);
    }

    GammaOutput &
    operator*() noexcept
    {
      return *output_;
    }

    GammaOutput *
    operator->() noexcept
    {
      return output_;
    }

  private:
    GammaOutput * output_;
  };

public:
  static GammaNode *
  create(jlm::rvsdg::output * predicate, size_t nalternatives)
  {
    return new GammaNode(predicate, nalternatives);
  }

  inline GammaInput *
  predicate() const noexcept;

  inline size_t
  nentryvars() const noexcept
  {
    JLM_ASSERT(node::ninputs() != 0);
    return node::ninputs() - 1;
  }

  inline size_t
  nexitvars() const noexcept
  {
    return node::noutputs();
  }

  inline GammaInput *
  entryvar(size_t index) const noexcept;

  [[nodiscard]] inline GammaOutput *
  exitvar(size_t index) const noexcept;

  inline GammaNode::entryvar_iterator
  begin_entryvar() const
  {
    if (nentryvars() == 0)
      return entryvar_iterator(nullptr);

    return entryvar_iterator(entryvar(0));
  }

  inline GammaNode::entryvar_iterator
  end_entryvar() const
  {
    return entryvar_iterator(nullptr);
  }

  inline GammaNode::exitvar_iterator
  begin_exitvar() const
  {
    if (nexitvars() == 0)
      return exitvar_iterator(nullptr);

    return exitvar_iterator(exitvar(0));
  }

  inline GammaNode::exitvar_iterator
  end_exitvar() const
  {
    return exitvar_iterator(nullptr);
  }

  inline GammaInput *
  add_entryvar(jlm::rvsdg::output * origin);

  inline GammaOutput *
  add_exitvar(const std::vector<jlm::rvsdg::output *> & values);

  /**
   * Removes all gamma outputs and their respective results. The outputs must have no users and
   * match the condition specified by \p match.
   *
   * @tparam F A type that supports the function call operator: bool operator(const GammaOutput&)
   * @param match Defines the condition of the elements to remove.
   */
  template<typename F>
  void
  RemoveGammaOutputsWhere(const F & match);

  /**
   * Removes all outputs that have no users.
   */
  void
  PruneOutputs()
  {
    auto match = [](const GammaOutput &)
    {
      return true;
    };

    RemoveGammaOutputsWhere(match);
  }

  virtual GammaNode *
  copy(jlm::rvsdg::Region * region, SubstitutionMap & smap) const override;
};

/* gamma input */

class GammaInput final : public structural_input
{
  friend GammaNode;

public:
  ~GammaInput() noexcept override;

private:
  GammaInput(GammaNode * node, jlm::rvsdg::output * origin, std::shared_ptr<const rvsdg::Type> type)
      : structural_input(node, origin, std::move(type))
  {}

public:
  GammaNode *
  node() const noexcept
  {
    return static_cast<GammaNode *>(structural_input::node());
  }

  inline argument_list::iterator
  begin()
  {
    return arguments.begin();
  }

  inline argument_list::const_iterator
  begin() const
  {
    return arguments.begin();
  }

  inline argument_list::iterator
  end()
  {
    return arguments.end();
  }

  inline argument_list::const_iterator
  end() const
  {
    return arguments.end();
  }

  inline size_t
  narguments() const noexcept
  {
    return arguments.size();
  }

  [[nodiscard]] RegionArgument *
  argument(size_t n) const noexcept
  {
    JLM_ASSERT(n < narguments());
    auto argument = node()->subregion(n)->argument(index() - 1);
    JLM_ASSERT(argument->input() == this);
    return argument;
  }
};

/* gamma output */

class GammaOutput final : public StructuralOutput
{
  friend GammaNode;

public:
  ~GammaOutput() noexcept override;

  GammaOutput(GammaNode * node, std::shared_ptr<const rvsdg::Type> type)
      : StructuralOutput(node, std::move(type))
  {}

  GammaNode *
  node() const noexcept
  {
    return static_cast<GammaNode *>(StructuralOutput::node());
  }

  inline result_list::iterator
  begin()
  {
    return results.begin();
  }

  inline result_list::const_iterator
  begin() const
  {
    return results.begin();
  }

  inline result_list::iterator
  end()
  {
    return results.end();
  }

  inline result_list::const_iterator
  end() const
  {
    return results.end();
  }

  inline size_t
  nresults() const noexcept
  {
    return results.size();
  }

  [[nodiscard]] RegionResult *
  result(size_t n) const noexcept
  {
    JLM_ASSERT(n < nresults());
    auto result = node()->subregion(n)->result(index());
    JLM_ASSERT(result->output() == this);
    return result;
  }

  /**
   * Determines whether a gamma output is invariant.
   *
   * A gamma output is invariant if its value directly originates from gamma inputs and the origin
   * of all these inputs is the same.
   *
   * @param invariantOrigin The origin of the gamma inputs if the gamma output is invariant and \p
   * invariantOrigin is unequal NULL.
   * @return True if the gamma output is invariant, otherwise false.
   */
  bool
  IsInvariant(rvsdg::output ** invariantOrigin = nullptr) const noexcept;
};

/* gamma node method definitions */

inline GammaNode::GammaNode(rvsdg::output * predicate, size_t nalternatives)
    : StructuralNode(GammaOperation(nalternatives), predicate->region(), nalternatives)
{
  node::add_input(std::unique_ptr<node_input>(
      new GammaInput(this, predicate, ControlType::Create(nalternatives))));
}

/**
 * Represents a region argument in a gamma subregion.
 */
class GammaArgument final : public RegionArgument
{
  friend GammaNode;

public:
  ~GammaArgument() noexcept override;

  GammaArgument &
  Copy(rvsdg::Region & region, structural_input * input) override;

private:
  GammaArgument(rvsdg::Region & region, GammaInput & input)
      : RegionArgument(&region, &input, input.Type())
  {}

  static GammaArgument &
  Create(rvsdg::Region & region, GammaInput & input)
  {
    auto gammaArgument = new GammaArgument(region, input);
    region.append_argument(gammaArgument);
    return *gammaArgument;
  }
};

/**
 * Represents a region result in a gamma subregion.
 */
class GammaResult final : public RegionResult
{
  friend GammaNode;

public:
  ~GammaResult() noexcept override;

private:
  GammaResult(rvsdg::Region & region, rvsdg::output & origin, GammaOutput & gammaOutput)
      : RegionResult(&region, &origin, &gammaOutput, origin.Type())
  {}

  GammaResult &
  Copy(rvsdg::output & origin, StructuralOutput * output) override;

  static GammaResult &
  Create(rvsdg::Region & region, rvsdg::output & origin, GammaOutput & gammaOutput)
  {
    auto gammaResult = new GammaResult(region, origin, gammaOutput);
    origin.region()->append_result(gammaResult);
    return *gammaResult;
  }
};

inline GammaInput *
GammaNode::predicate() const noexcept
{
  return util::AssertedCast<GammaInput>(StructuralNode::input(0));
}

inline GammaInput *
GammaNode::entryvar(size_t index) const noexcept
{
  return util::AssertedCast<GammaInput>(node::input(index + 1));
}

inline GammaOutput *
GammaNode::exitvar(size_t index) const noexcept
{
  return static_cast<GammaOutput *>(node::output(index));
}

inline GammaInput *
GammaNode::add_entryvar(jlm::rvsdg::output * origin)
{
  auto input =
      node::add_input(std::unique_ptr<node_input>(new GammaInput(this, origin, origin->Type())));
  auto gammaInput = util::AssertedCast<GammaInput>(input);

  for (size_t n = 0; n < nsubregions(); n++)
  {
    GammaArgument::Create(*subregion(n), *gammaInput);
  }

  return gammaInput;
}

inline GammaOutput *
GammaNode::add_exitvar(const std::vector<jlm::rvsdg::output *> & values)
{
  if (values.size() != nsubregions())
    throw jlm::util::error("Incorrect number of values.");

  const auto & type = values[0]->Type();
  node::add_output(std::make_unique<GammaOutput>(this, type));

  auto output = exitvar(nexitvars() - 1);
  for (size_t n = 0; n < nsubregions(); n++)
  {
    GammaResult::Create(*subregion(n), *values[n], *output);
  }

  return output;
}

template<typename F>
void
GammaNode::RemoveGammaOutputsWhere(const F & match)
{
  // iterate backwards to avoid the invalidation of 'n' by RemoveOutput()
  for (size_t n = noutputs() - 1; n != static_cast<size_t>(-1); n--)
  {
    auto & gammaOutput = *util::AssertedCast<const GammaOutput>(output(n));
    if (gammaOutput.nusers() == 0 && match(gammaOutput))
    {
      for (size_t r = 0; r < nsubregions(); r++)
      {
        subregion(r)->RemoveResult(n);
      }

      RemoveOutput(n);
    }
  }
}

}

#endif

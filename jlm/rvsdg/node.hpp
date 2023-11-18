/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_NODE_HPP
#define JLM_RVSDG_NODE_HPP

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <unordered_set>
#include <utility>

#include <jlm/rvsdg/operation.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/intrusive-list.hpp>
#include <jlm/util/strfmt.hpp>

namespace jlm::rvsdg
{
namespace base
{
class type;
}

class graph;
class node_normal_form;
class output;
class substitution_map;

/* inputs */

class input
{
  friend jlm::rvsdg::node;
  friend jlm::rvsdg::region;

public:
  virtual ~input() noexcept;

  input(jlm::rvsdg::output * origin, jlm::rvsdg::region * region, const jlm::rvsdg::port & port);

  input(const input &) = delete;

  input(input &&) = delete;

  input &
  operator=(const input &) = delete;

  input &
  operator=(input &&) = delete;

  inline size_t
  index() const noexcept
  {
    return index_;
  }

  jlm::rvsdg::output *
  origin() const noexcept
  {
    return origin_;
  }

  void
  divert_to(jlm::rvsdg::output * new_origin);

  inline const jlm::rvsdg::type &
  type() const noexcept
  {
    return port_->type();
  }

  inline jlm::rvsdg::region *
  region() const noexcept
  {
    return region_;
  }

  inline const jlm::rvsdg::port &
  port() const noexcept
  {
    return *port_;
  }

  virtual std::string
  debug_string() const;

  inline void
  replace(const jlm::rvsdg::port & port)
  {
    if (port_->type() != port.type())
      throw jlm::util::type_error(port_->type().debug_string(), port.type().debug_string());

    port_ = port.copy();
  }

  /**
   * Retrieve the associated node from \p input if \p input is derived from jlm::rvsdg::node_input.
   *
   * @param input The input from which to retrieve the node.
   * @return The node associated with \p input if input is derived from jlm::rvsdg::node_input,
   * otherwise nullptr.
   */
  [[nodiscard]] static jlm::rvsdg::node *
  GetNode(const jlm::rvsdg::input & input) noexcept;

  template<class T>
  class iterator
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T *;
    using difference_type = std::ptrdiff_t;
    using pointer = T **;
    using reference = T *&;

    static_assert(
        std::is_base_of<jlm::rvsdg::input, T>::value,
        "Template parameter T must be derived from jlm::rvsdg::input.");

  protected:
    constexpr iterator(T * value)
        : value_(value)
    {}

    virtual T *
    next() const
    {
      /*
        I cannot make this method abstract due to the return value of operator++(int).
        This is the best I could come up with as a workaround.
      */
      throw jlm::util::error("This method must be overloaded.");
    }

  public:
    T *
    value() const noexcept
    {
      return value_;
    }

    T &
    operator*()
    {
      JLM_ASSERT(value_ != nullptr);
      return *value_;
    }

    T *
    operator->() const
    {
      return value_;
    }

    iterator<T> &
    operator++()
    {
      value_ = next();
      return *this;
    }

    iterator<T>
    operator++(int)
    {
      iterator<T> tmp = *this;
      ++*this;
      return tmp;
    }

    virtual bool
    operator==(const iterator<T> & other) const
    {
      return value_ == other.value_;
    }

    bool
    operator!=(const iterator<T> & other) const
    {
      return !operator==(other);
    }

  private:
    T * value_;
  };

  template<class T>
  class constiterator
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const T *;
    using difference_type = std::ptrdiff_t;
    using pointer = const T **;
    using reference = const T *&;

    static_assert(
        std::is_base_of<jlm::rvsdg::input, T>::value,
        "Template parameter T must be derived from jlm::rvsdg::input.");

  protected:
    constexpr constiterator(const T * value)
        : value_(value)
    {}

    virtual const T *
    next() const
    {
      /*
        I cannot make this method abstract due to the return value of operator++(int).
        This is the best I could come up with as a workaround.
      */
      throw jlm::util::error("This method must be overloaded.");
    }

  public:
    const T *
    value() const noexcept
    {
      return value_;
    }

    const T &
    operator*()
    {
      JLM_ASSERT(value_ != nullptr);
      return *value_;
    }

    const T *
    operator->() const
    {
      return value_;
    }

    constiterator<T> &
    operator++()
    {
      value_ = next();
      return *this;
    }

    constiterator<T>
    operator++(int)
    {
      constiterator<T> tmp = *this;
      ++*this;
      return tmp;
    }

    virtual bool
    operator==(const constiterator<T> & other) const
    {
      return value_ == other.value_;
    }

    bool
    operator!=(const constiterator<T> & other) const
    {
      return !operator==(other);
    }

  private:
    const T * value_;
  };

private:
  size_t index_;
  jlm::rvsdg::output * origin_;
  jlm::rvsdg::region * region_;
  std::unique_ptr<jlm::rvsdg::port> port_;
};

template<class T>
static inline bool
is(const jlm::rvsdg::input & input) noexcept
{
  static_assert(
      std::is_base_of<jlm::rvsdg::input, T>::value,
      "Template parameter T must be derived from jlm::rvsdg::input.");

  return dynamic_cast<const T *>(&input) != nullptr;
}

/* outputs */

class output
{
  friend input;
  friend jlm::rvsdg::node;
  friend jlm::rvsdg::region;

  typedef std::unordered_set<jlm::rvsdg::input *>::const_iterator user_iterator;

public:
  virtual ~output() noexcept;

  output(jlm::rvsdg::region * region, const jlm::rvsdg::port & port);

  output(const output &) = delete;

  output(output &&) = delete;

  output &
  operator=(const output &) = delete;

  output &
  operator=(output &&) = delete;

  inline size_t
  index() const noexcept
  {
    return index_;
  }

  inline size_t
  nusers() const noexcept
  {
    return users_.size();
  }

  inline void
  divert_users(jlm::rvsdg::output * new_origin)
  {
    if (this == new_origin)
      return;

    while (users_.size())
      (*users_.begin())->divert_to(new_origin);
  }

  inline user_iterator
  begin() const noexcept
  {
    return users_.begin();
  }

  inline user_iterator
  end() const noexcept
  {
    return users_.end();
  }

  inline const jlm::rvsdg::type &
  type() const noexcept
  {
    return port_->type();
  }

  inline jlm::rvsdg::region *
  region() const noexcept
  {
    return region_;
  }

  inline const jlm::rvsdg::port &
  port() const noexcept
  {
    return *port_;
  }

  virtual std::string
  debug_string() const;

  inline void
  replace(const jlm::rvsdg::port & port)
  {
    if (port_->type() != port.type())
      throw jlm::util::type_error(port_->type().debug_string(), port.type().debug_string());

    port_ = port.copy();
  }

  template<class T>
  class iterator
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T *;
    using difference_type = std::ptrdiff_t;
    using pointer = T **;
    using rerefence = T *&;

    static_assert(
        std::is_base_of<jlm::rvsdg::output, T>::value,
        "Template parameter T must be derived from jlm::rvsdg::output.");

  protected:
    constexpr iterator(T * value)
        : value_(value)
    {}

    virtual T *
    next() const
    {
      /*
        I cannot make this method abstract due to the return value of operator++(int).
        This is the best I could come up with as a workaround.
      */
      throw jlm::util::error("This method must be overloaded.");
    }

  public:
    T *
    value() const noexcept
    {
      return value_;
    }

    T &
    operator*()
    {
      JLM_ASSERT(value_ != nullptr);
      return *value_;
    }

    T *
    operator->() const
    {
      return value_;
    }

    iterator<T> &
    operator++()
    {
      value_ = next();
      return *this;
    }

    iterator<T>
    operator++(int)
    {
      iterator<T> tmp = *this;
      ++*this;
      return tmp;
    }

    virtual bool
    operator==(const iterator<T> & other) const
    {
      return value_ == other.value_;
    }

    bool
    operator!=(const iterator<T> & other) const
    {
      return !operator==(other);
    }

  private:
    T * value_;
  };

  template<class T>
  class constiterator
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const T *;
    using difference_type = std::ptrdiff_t;
    using pointer = const T **;
    using reference = const T *&;

    static_assert(
        std::is_base_of<jlm::rvsdg::output, T>::value,
        "Template parameter T must be derived from jlm::rvsdg::output.");

  protected:
    constexpr constiterator(const T * value)
        : value_(value)
    {}

    virtual const T *
    next() const
    {
      /*
        I cannot make this method abstract due to the return value of operator++(int).
        This is the best I could come up with as a workaround.
      */
      throw jlm::util::error("This method must be overloaded.");
    }

  public:
    const T *
    value() const noexcept
    {
      return value_;
    }

    const T &
    operator*()
    {
      JLM_ASSERT(value_ != nullptr);
      return *value_;
    }

    const T *
    operator->() const
    {
      return value_;
    }

    constiterator<T> &
    operator++()
    {
      value_ = next();
      return *this;
    }

    constiterator<T>
    operator++(int)
    {
      constiterator<T> tmp = *this;
      ++*this;
      return tmp;
    }

    virtual bool
    operator==(const constiterator<T> & other) const
    {
      return value_ == other.value_;
    }

    bool
    operator!=(const constiterator<T> & other) const
    {
      return !operator==(other);
    }

  private:
    const T * value_;
  };

private:
  void
  remove_user(jlm::rvsdg::input * user);

  void
  add_user(jlm::rvsdg::input * user);

  size_t index_;
  jlm::rvsdg::region * region_;
  std::unique_ptr<jlm::rvsdg::port> port_;
  std::unordered_set<jlm::rvsdg::input *> users_;
};

template<class T>
static inline bool
is(const jlm::rvsdg::output * output) noexcept
{
  static_assert(
      std::is_base_of<jlm::rvsdg::output, T>::value,
      "Template parameter T must be derived from jlm::rvsdg::output.");

  return dynamic_cast<const T *>(output) != nullptr;
}

/* node_input class */

class node_input : public jlm::rvsdg::input
{
public:
  node_input(jlm::rvsdg::output * origin, jlm::rvsdg::node * node, const jlm::rvsdg::port & port);

  jlm::rvsdg::node *
  node() const noexcept
  {
    return node_;
  }

  /**
   * Returns the associated node if \p input is a jlm::rvsdg::node_input, otherwise null.
   *
   * @param input A jlm::rvsdg::input
   * @return Returns a jlm::rvsdg::node or null.
   *
   * @see jlm::rvsdg::node_input::node()
   */
  [[nodiscard]] static jlm::rvsdg::node *
  node(const jlm::rvsdg::input & input)
  {
    auto nodeInput = dynamic_cast<const node_input *>(&input);
    return nodeInput != nullptr ? nodeInput->node() : nullptr;
  }

private:
  jlm::rvsdg::node * node_;
};

/* node_output class */

class node_output : public jlm::rvsdg::output
{
public:
  node_output(jlm::rvsdg::node * node, const jlm::rvsdg::port & port);

  jlm::rvsdg::node *
  node() const noexcept
  {
    return node_;
  }

  static jlm::rvsdg::node *
  node(const jlm::rvsdg::output * output)
  {
    auto no = dynamic_cast<const node_output *>(output);
    return no != nullptr ? no->node() : nullptr;
  }

private:
  jlm::rvsdg::node * node_;
};

/* node class */

class node
{
public:
  virtual ~node();

  node(std::unique_ptr<jlm::rvsdg::operation> op, jlm::rvsdg::region * region);

  inline const jlm::rvsdg::operation &
  operation() const noexcept
  {
    return *operation_;
  }

  inline bool
  has_users() const noexcept
  {
    for (const auto & output : outputs_)
    {
      if (output->nusers() != 0)
        return true;
    }

    return false;
  }

  inline bool
  has_predecessors() const noexcept
  {
    for (const auto & input : inputs_)
    {
      if (is<node_output>(input->origin()))
        return true;
    }

    return false;
  }

  inline bool
  has_successors() const noexcept
  {
    for (const auto & output : outputs_)
    {
      for (const auto & user : *output)
      {
        if (is<node_input>(*user))
          return true;
      }
    }

    return false;
  }

  inline size_t
  ninputs() const noexcept
  {
    return inputs_.size();
  }

  node_input *
  input(size_t index) const noexcept
  {
    JLM_ASSERT(index < ninputs());
    return inputs_[index].get();
  }

  inline size_t
  noutputs() const noexcept
  {
    return outputs_.size();
  }

  node_output *
  output(size_t index) const noexcept
  {
    JLM_ASSERT(index < noutputs());
    return outputs_[index].get();
  }

  inline void
  recompute_depth() noexcept;

protected:
  node_input *
  add_input(std::unique_ptr<node_input> input);

  /**
   * Removes an input from the node given the inputs' index.
   *
   * The removal of an input invalidates the node's existing input iterators.
   *
   * @param index The inputs' index. It must be between [0, ninputs()).
   *
   * \note The method must adjust the indices of the other inputs after the removal. Moreover, it
   * also might need to recompute the depth of the node.
   *
   * \see ninputs()
   * \see recompute_depth()
   * \see input#index()
   */
  void
  RemoveInput(size_t index);

  /**
   * Removes all inputs that match the condition specified by \p match.
   *
   * @tparam F A type that supports the function call operator: bool operator(const node_input&)
   * @param match Defines the condition for the inputs to remove.
   */
  template<typename F>
  void
  RemoveInputsWhere(const F & match)
  {
    // iterate backwards to avoid the invalidation of 'n' by RemoveInput()
    for (size_t n = ninputs() - 1; n != static_cast<size_t>(-1); n--)
    {
      auto & input = *node::input(n);
      if (match(input))
      {
        RemoveInput(n);
      }
    }
  }

  node_output *
  add_output(std::unique_ptr<node_output> output)
  {
    output->index_ = noutputs();
    outputs_.push_back(std::move(output));
    return this->output(noutputs() - 1);
  }

  /**
   * Removes an output from the node given the outputs' index.
   *
   * An output can only be removed, if it has no users. The removal of an output invalidates the
   * node's existing output iterators.
   *
   * @param index The outputs' index. It must be between [0, noutputs()).
   *
   * \note The method must adjust the indices of the other outputs after the removal. The methods'
   * runtime is therefore O(n), where n is the node's number of outputs.
   *
   * \see noutputs()
   * \see output#index()
   * \see output#nusers()
   */
  void
  RemoveOutput(size_t index);

  /**
   * Removes all outputs that have no users and match the condition specified by \p match.
   *
   * @tparam F A type that supports the function call operator: bool operator(const node_output&)
   * @param match Defines the condition for the outputs to remove.
   *
   * \see output#nusers()
   */
  template<typename F>
  void
  RemoveOutputsWhere(const F & match)
  {
    // iterate backwards to avoid the invalidation of 'n' by RemoveOutput()
    for (size_t n = noutputs() - 1; n != static_cast<size_t>(-1); n--)
    {
      auto & output = *node::output(n);
      if (output.nusers() == 0 && match(output))
      {
        RemoveOutput(n);
      }
    }
  }

public:
  inline jlm::rvsdg::graph *
  graph() const noexcept
  {
    return graph_;
  }

  inline jlm::rvsdg::region *
  region() const noexcept
  {
    return region_;
  }

  virtual jlm::rvsdg::node *
  copy(jlm::rvsdg::region * region, const std::vector<jlm::rvsdg::output *> & operands) const;

  /**
    \brief Copy a node with substitutions
    \param region Target region to create node in
    \param smap Operand substitutions
    \return Copied node

    Create a new node that is semantically equivalent to an
    existing node. The newly created node will use the same
    operands as the existing node unless there is a substitution
    registered for a particular operand.

    The given substitution map is updated so that all
    outputs of the original node will be substituted by
    corresponding outputs of the newly created node in
    subsequent \ref copy operations.
  */
  virtual jlm::rvsdg::node *
  copy(jlm::rvsdg::region * region, jlm::rvsdg::substitution_map & smap) const = 0;

  inline size_t
  depth() const noexcept
  {
    return depth_;
  }

private:
  jlm::util::intrusive_list_anchor<jlm::rvsdg::node> region_node_list_anchor_;

  jlm::util::intrusive_list_anchor<jlm::rvsdg::node> region_top_node_list_anchor_;

  jlm::util::intrusive_list_anchor<jlm::rvsdg::node> region_bottom_node_list_anchor_;

public:
  typedef jlm::util::
      intrusive_list_accessor<jlm::rvsdg::node, &jlm::rvsdg::node::region_node_list_anchor_>
          region_node_list_accessor;

  typedef jlm::util::
      intrusive_list_accessor<jlm::rvsdg::node, &jlm::rvsdg::node::region_top_node_list_anchor_>
          region_top_node_list_accessor;

  typedef jlm::util::
      intrusive_list_accessor<jlm::rvsdg::node, &jlm::rvsdg::node::region_bottom_node_list_anchor_>
          region_bottom_node_list_accessor;

private:
  size_t depth_;
  jlm::rvsdg::graph * graph_;
  jlm::rvsdg::region * region_;
  std::unique_ptr<jlm::rvsdg::operation> operation_;
  std::vector<std::unique_ptr<node_input>> inputs_;
  std::vector<std::unique_ptr<node_output>> outputs_;
};

static inline std::vector<jlm::rvsdg::output *>
operands(const jlm::rvsdg::node * node)
{
  std::vector<jlm::rvsdg::output *> operands;
  for (size_t n = 0; n < node->ninputs(); n++)
    operands.push_back(node->input(n)->origin());
  return operands;
}

static inline std::vector<jlm::rvsdg::output *>
outputs(const jlm::rvsdg::node * node)
{
  std::vector<jlm::rvsdg::output *> outputs;
  for (size_t n = 0; n < node->noutputs(); n++)
    outputs.push_back(node->output(n));
  return outputs;
}

static inline void
divert_users(jlm::rvsdg::node * node, const std::vector<jlm::rvsdg::output *> & outputs)
{
  JLM_ASSERT(node->noutputs() == outputs.size());

  for (size_t n = 0; n < outputs.size(); n++)
    node->output(n)->divert_users(outputs[n]);
}

template<class T>
static inline bool
is(const jlm::rvsdg::node * node) noexcept
{
  if (!node)
    return false;

  return is<T>(node->operation());
}

jlm::rvsdg::node *
producer(const jlm::rvsdg::output * output) noexcept;

bool
normalize(jlm::rvsdg::node * node);

}

#endif

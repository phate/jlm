/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_NODE_HPP
#define JLM_RVSDG_NODE_HPP

#include <jlm/rvsdg/operation.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/HashSet.hpp>
#include <jlm/util/intrusive-list.hpp>
#include <jlm/util/iterator_range.hpp>
#include <jlm/util/IteratorWrapper.hpp>

#include <cstdint>
#include <utility>
#include <variant>

namespace jlm::rvsdg
{

class Graph;
class Output;
class SubstitutionMap;

class Input
{
  friend class Node;
  friend class Region;

protected:
  Input(Node & owner, Output & origin, std::shared_ptr<const rvsdg::Type> type);

  Input(Region & owner, Output & origin, std::shared_ptr<const rvsdg::Type> type);

public:
  virtual ~Input() noexcept;

  Input(const Input &) = delete;

  Input(Input &&) = delete;

  Input &
  operator=(const Input &) = delete;

  Input &
  operator=(Input &&) = delete;

  size_t
  index() const noexcept
  {
    return index_;
  }

  Output *
  origin() const noexcept
  {
    return origin_;
  }

  void
  divert_to(Output * new_origin);

  [[nodiscard]] const std::shared_ptr<const rvsdg::Type> &
  Type() const noexcept
  {
    return Type_;
  }

  [[nodiscard]] Region *
  region() const noexcept;

  virtual std::string
  debug_string() const;

  [[nodiscard]] std::variant<Node *, Region *>
  GetOwner() const noexcept
  {
    return Owner_;
  }

  class Iterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Input;
    using difference_type = std::ptrdiff_t;
    using pointer = Input *;
    using reference = Input &;

    constexpr explicit Iterator(Input * input)
        : Input_(input)
    {}

    [[nodiscard]] Input *
    GetInput() const noexcept
    {
      return Input_;
    }

    Input &
    operator*()
    {
      JLM_ASSERT(Input_ != nullptr);
      return *Input_;
    }

    Input *
    operator->() const
    {
      return Input_;
    }

    Iterator &
    operator++()
    {
      Input_ = ComputeNext();
      return *this;
    }

    Iterator
    operator++(int)
    {
      Iterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool
    operator==(const Iterator & other) const
    {
      return Input_ == other.Input_;
    }

    bool
    operator!=(const Iterator & other) const
    {
      return !operator==(other);
    }

  private:
    [[nodiscard]] Input *
    ComputeNext() const;

    Input * Input_;
  };

  class ConstIterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const Input;
    using difference_type = std::ptrdiff_t;
    using pointer = const Input *;
    using reference = const Input &;

    constexpr explicit ConstIterator(const Input * input)
        : Input_(input)
    {}

    [[nodiscard]] const Input *
    GetInput() const noexcept
    {
      return Input_;
    }

    const Input &
    operator*()
    {
      JLM_ASSERT(Input_ != nullptr);
      return *Input_;
    }

    const Input *
    operator->() const
    {
      return Input_;
    }

    ConstIterator &
    operator++()
    {
      Input_ = ComputeNext();
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
      return Input_ == other.Input_;
    }

    bool
    operator!=(const ConstIterator & other) const
    {
      return !operator==(other);
    }

  private:
    [[nodiscard]] Input *
    ComputeNext() const;

    const Input * Input_;
  };

private:
  static void
  CheckTypes(
      const Region & region,
      const Output & origin,
      const std::shared_ptr<const rvsdg::Type> & type);

  size_t index_;
  jlm::rvsdg::Output * origin_ = nullptr;
  std::variant<Node *, Region *> Owner_;
  std::shared_ptr<const rvsdg::Type> Type_;
  jlm::util::IntrusiveListAnchor<Input> UsersList_;
  using UsersListAccessor = util::IntrusiveListAccessor<Input, &Input::UsersList_>;
  using UsersList = jlm::util::IntrusiveList<Input, UsersListAccessor>;

  friend class Output;
};

template<class T>
static inline bool
is(const jlm::rvsdg::Input & input) noexcept
{
  static_assert(
      std::is_base_of<jlm::rvsdg::Input, T>::value,
      "Template parameter T must be derived from jlm::rvsdg::input.");

  return dynamic_cast<const T *>(&input) != nullptr;
}

class Output
{
  friend Input;
  friend class Node;
  friend class Region;

protected:
  Output(Node & owner, std::shared_ptr<const rvsdg::Type> type);

  Output(Region * owner, std::shared_ptr<const rvsdg::Type> type);

public:
  using UsersList = Input::UsersList;
  using UsersRange = jlm::util::IteratorRange<UsersList::Iterator>;
  using UsersConstRange = jlm::util::IteratorRange<UsersList::ConstIterator>;

  virtual ~Output() noexcept;

  Output(const Output &) = delete;

  Output(Output &&) = delete;

  Output &
  operator=(const Output &) = delete;

  Output &
  operator=(Output &&) = delete;

  size_t
  index() const noexcept
  {
    return index_;
  }

  size_t
  nusers() const noexcept
  {
    return NumUsers_;
  }

  /**
   * Determines whether the output is dead.
   *
   * An output is considered dead when it has no users.
   *
   * @return True, if the output is dead, otherwise false.
   *
   * \see nusers()
   */
  [[nodiscard]] bool
  IsDead() const noexcept
  {
    return NumUsers_ == 0;
  }

  inline void
  divert_users(jlm::rvsdg::Output * new_origin)
  {
    if (this == new_origin)
      return;

    while (!Users_.empty())
      Users_.begin()->divert_to(new_origin);
  }

  /**
   * Divert all users of the output that satisfy the predicate \p match.
   *
   * @tparam F A functor with the signature (const rvsdg::Input &) -> bool
   * @param newOrigin The new origin of each user that satisfies \p match.
   * @param match An instance of F, to be invoked on each user
   *
   * @return The number of diverted users.
   */
  template<typename F>
  size_t
  divertUsersWhere(Output & newOrigin, const F & match)
  {
    if (this == &newOrigin)
      return 0;

    util::HashSet<Input *> matchedUsers;
    for (auto & user : Users_)
    {
      if (match(user))
        matchedUsers.insert(&user);
    }

    for (auto & user : matchedUsers.Items())
    {
      user->divert_to(&newOrigin);
    }

    return matchedUsers.Size();
  }

  /**
   * @return The first and only user of the output.
   *
   * \pre The output has only a single user.
   */
  [[nodiscard]] rvsdg::Input &
  SingleUser() noexcept
  {
    JLM_ASSERT(NumUsers_ == 1);
    return *Users_.begin();
  }

  UsersRange
  Users()
  {
    return UsersRange(Users_.begin(), Users_.end());
  }

  UsersConstRange
  Users() const
  {
    return UsersConstRange(Users_.cbegin(), Users_.cend());
  }

  [[nodiscard]] const std::shared_ptr<const rvsdg::Type> &
  Type() const noexcept
  {
    return Type_;
  }

  [[nodiscard]] rvsdg::Region *
  region() const noexcept;

  virtual std::string
  debug_string() const;

  [[nodiscard]] std::variant<Node *, Region *>
  GetOwner() const noexcept
  {
    return Owner_;
  }

  class Iterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Output;
    using difference_type = std::ptrdiff_t;
    using pointer = Output *;
    using reference = Output &;

    constexpr explicit Iterator(Output * output)
        : Output_(output)
    {}

    [[nodiscard]] Output *
    GetOutput() const noexcept
    {
      return Output_;
    }

    Output &
    operator*()
    {
      JLM_ASSERT(Output_ != nullptr);
      return *Output_;
    }

    Output *
    operator->() const
    {
      return Output_;
    }

    Iterator &
    operator++()
    {
      Output_ = ComputeNext();
      return *this;
    }

    Iterator
    operator++(int)
    {
      Iterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool
    operator==(const Iterator & other) const
    {
      return Output_ == other.Output_;
    }

    bool
    operator!=(const Iterator & other) const
    {
      return !operator==(other);
    }

  private:
    [[nodiscard]] Output *
    ComputeNext() const;

    Output * Output_;
  };

  class ConstIterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const Output;
    using difference_type = std::ptrdiff_t;
    using pointer = const Output *;
    using reference = const Output &;

    constexpr explicit ConstIterator(const Output * output)
        : Output_(output)
    {}

    const Output *
    GetOutput() const noexcept
    {
      return Output_;
    }

    const Output &
    operator*() const
    {
      JLM_ASSERT(Output_ != nullptr);
      return *Output_;
    }

    const Output *
    operator->() const
    {
      return Output_;
    }

    ConstIterator &
    operator++()
    {
      Output_ = ComputeNext();
      return *this;
    }

    ConstIterator
    operator++(int)
    {
      const ConstIterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool
    operator==(const ConstIterator & other) const
    {
      return Output_ == other.Output_;
    }

    bool
    operator!=(const ConstIterator & other) const
    {
      return !operator==(other);
    }

  private:
    [[nodiscard]] Output *
    ComputeNext() const;

    const Output * Output_;
  };

private:
  void
  remove_user(jlm::rvsdg::Input * user);

  void
  add_user(jlm::rvsdg::Input * user);

  size_t index_;
  std::variant<Node *, Region *> Owner_;
  std::shared_ptr<const rvsdg::Type> Type_;
  UsersList Users_;
  std::size_t NumUsers_ = 0;
};

/**
 * Routes \p output through the region tree to \p region by creating the necessary inputs and region
 * arguments for the encountered nodes.
 *
 * \note The function throws an exception if the region of \p output is not an ancestor of \p
 * region.
 *
 * @param output The value that is supposed to be routed to \p region.
 * @param region The region the value is supposed to be routed to.
 * @return The routed value in \p region.
 */
Output &
RouteToRegion(Output & output, Region & region);

template<class T>
static inline bool
is(const jlm::rvsdg::Output * output) noexcept
{
  static_assert(
      std::is_base_of<jlm::rvsdg::Output, T>::value,
      "Template parameter T must be derived from jlm::rvsdg::output.");

  return dynamic_cast<const T *>(output) != nullptr;
}

class NodeInput : public Input
{
public:
  NodeInput(Output * origin, Node * node, std::shared_ptr<const rvsdg::Type> type);

  [[nodiscard]] Node *
  node() const noexcept
  {
    return std::get<Node *>(GetOwner());
  }
};

class NodeOutput : public Output
{
public:
  NodeOutput(Node * node, std::shared_ptr<const rvsdg::Type> type);

  [[nodiscard]] Node *
  node() const noexcept
  {
    return std::get<Node *>(GetOwner());
  }
};

/* node class */

class Node
{
public:
  using Id = uint64_t;

  using InputIteratorRange = util::IteratorRange<Input::Iterator>;
  using InputConstIteratorRange = util::IteratorRange<Input::ConstIterator>;
  using OutputIteratorRange = util::IteratorRange<Output::Iterator>;
  using OutputConstIteratorRange = util::IteratorRange<Output::ConstIterator>;

  virtual ~Node();

  explicit Node(Region * region);

  /**
   * @return The unique identifier of the node instance within the region.
   *
   * \see Region::generateNodeId()
   */
  [[nodiscard]] Id
  GetNodeId() const noexcept
  {
    return Id_;
  }

  [[nodiscard]] virtual const Operation &
  GetOperation() const noexcept = 0;

  inline size_t
  ninputs() const noexcept
  {
    return inputs_.size();
  }

  NodeInput *
  input(size_t index) const noexcept
  {
    JLM_ASSERT(index < ninputs());
    return inputs_[index].get();
  }

  [[nodiscard]] InputIteratorRange
  Inputs() noexcept
  {
    if (ninputs() == 0)
    {
      return { Input::Iterator(nullptr), Input::Iterator(nullptr) };
    }

    return { Input::Iterator(input(0)), Input::Iterator(nullptr) };
  }

  [[nodiscard]] InputConstIteratorRange
  Inputs() const noexcept
  {
    if (ninputs() == 0)
    {
      return { Input::ConstIterator(nullptr), Input::ConstIterator(nullptr) };
    }

    return { Input::ConstIterator(input(0)), Input::ConstIterator(nullptr) };
  }

  inline size_t
  noutputs() const noexcept
  {
    return outputs_.size();
  }

  NodeOutput *
  output(size_t index) const noexcept
  {
    JLM_ASSERT(index < noutputs());
    return outputs_[index].get();
  }

  [[nodiscard]] OutputIteratorRange
  Outputs() noexcept
  {
    if (noutputs() == 0)
    {
      return { Output::Iterator(nullptr), Output::Iterator(nullptr) };
    }

    return { Output::Iterator(output(0)), Output::Iterator(nullptr) };
  }

  [[nodiscard]] OutputConstIteratorRange
  Outputs() const noexcept
  {
    if (noutputs() == 0)
    {
      return { Output::ConstIterator(nullptr), Output::ConstIterator(nullptr) };
    }

    return { Output::ConstIterator(output(0)), Output::ConstIterator(nullptr) };
  }

  /**
   * \brief Determines whether the node is dead.
   *
   * A node is considered dead if all its outputs are dead.
   *
   * @return True, if the node is dead, otherwise false.
   *
   * \see output::IsDead()
   */
  [[nodiscard]] bool
  IsDead() const noexcept
  {
    return numSuccessors_ == 0;
  }

  [[nodiscard]] std::size_t
  numSuccessors() const noexcept
  {
    return numSuccessors_;
  }

  virtual std::string
  DebugString() const = 0;

protected:
  /**
   * Adds the given \p input to the node's inputs.
   * Invalidates existing iterators to the node's inputs.
   *
   * @param input an owned pointer to the new input
   * @param notifyRegion If true, the region is informed about the new input.
   * This should be false if the node has not yet notified the region about being created,
   * i.e., this function is being called from the node's constructor.
   *
   * @return a pointer to the added input
   */
  NodeInput *
  addInput(std::unique_ptr<NodeInput> input, bool notifyRegion);

  /**
   * Removes an input from the node given the inputs' \p index.
   *
   * The removal of an input invalidates the node's existing input iterators,
   * and changes the index of all following inputs.
   *
   * @param index The inputs' index. It must be between [0, ninputs()).
   * @param notifyRegion If true, the region is informed about the removal.
   * This should be false if the node has already notified the region about being removed,
   * i.e., this function is being called from the node's destructor.
   *
   * \see ninputs()
   * \see input#index()
   */
  void
  removeInput(size_t index, bool notifyRegion);

  // FIXME: I really would not like to be RemoveInputsWhere() to be public
public:
  /**
   * Removes all inputs that have an index in \p indices.
   *
   * @param indices The indices of the arguments that should be removed.
   * @param notifyRegion If true, the region is informed about the removal of an input.
   * This should be false if the node has already notified the region about being removed,
   * i.e., this function is being called from the node's destructor.
   *
   * @return The number of inputs that were removed. This might be less than the number of indices
   * as some provided input indices might not belong to an actual input.
   */
  size_t
  RemoveInputs(const util::HashSet<size_t> & indices, bool notifyRegion);

protected:
  NodeOutput *
  addOutput(std::unique_ptr<NodeOutput> output)
  {
    if (output->node() != this)
      throw std::logic_error("Output does not belong to this node!");
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
  removeOutput(size_t index);

  // FIXME: I really would not like to be RemoveOutputsWhere() to be public
public:
  /**
   * Removes all outputs that have no users and an index contained in \p indices.
   *
   * @param indices The indices of the outputs that should be removed.
   *
   * @return The number of outputs that were actually removed. This might be less than the number
   * of indices as some outputs might not have been dead or a provided output index does not
   * belong to an output argument.
   *
   * \see output#nusers()
   */
  size_t
  RemoveOutputs(const util::HashSet<size_t> & indices);

  [[nodiscard]] Graph *
  graph() const noexcept;

  [[nodiscard]] rvsdg::Region *
  region() const noexcept
  {
    return region_;
  }

  virtual Node *
  copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::Output *> & operands) const;

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
  virtual Node *
  copy(rvsdg::Region * region, SubstitutionMap & smap) const = 0;

private:
  util::IntrusiveListAnchor<Node> region_node_list_anchor_{};

  util::IntrusiveListAnchor<Node> region_top_node_list_anchor_{};

  util::IntrusiveListAnchor<Node> region_bottom_node_list_anchor_{};

public:
  typedef util::IntrusiveListAccessor<Node, &Node::region_node_list_anchor_>
      region_node_list_accessor;

  typedef util::IntrusiveListAccessor<Node, &Node::region_top_node_list_anchor_>
      region_top_node_list_accessor;

  typedef util::IntrusiveListAccessor<Node, &Node::region_bottom_node_list_anchor_>
      region_bottom_node_list_accessor;

private:
  Id Id_;
  Region * region_;
  std::vector<std::unique_ptr<NodeInput>> inputs_;
  std::vector<std::unique_ptr<NodeOutput>> outputs_;
  std::size_t numSuccessors_ = 0;

  friend class Output;
};

/**
 * \brief Checks if the given node is not null, and has an operation of the specified type.
 *
 * \tparam OperationType
 *   The subclass of operation to check for.
 *
 * \param node
 *   The node being checked.
 *
 * \returns
 *   true if node has an operation that is an instance of the specified type, otherwise false.
 */
template<class OperationType>
inline bool
is(const Node * node) noexcept
{
  if (!node)
    return false;

  return is<OperationType>(node->GetOperation());
}

/**
 * Attempts to get the operation of the given node, if the operation is of the given type.
 * @tparam TOperation the type of the operation
 * @param node the node
 * @return the node's operation, or nullptr if the node has an operation of the wrong type.
 */
template<typename TOperation>
[[nodiscard]] const TOperation *
tryGetOperation(const Node & node) noexcept
{
  return dynamic_cast<const TOperation *>(&node.GetOperation());
}

/**
 * \brief Checks if this is an input to a node of specified type.
 *
 * \tparam NodeType
 *   The node type to be matched against.
 *
 * \param input
 *   Input to be checked.
 *
 * \returns
 *   Owning node of requested type or nullptr.
 *
 * Checks if the specified input belongs to a node of requested type.
 * If this is the case, returns a pointer to the node of matched type.
 * If this is not the case (because either this as a region exit
 * result or its owning node is not of the requested type), returns
 * nullptr.
 *
 * See \ref def_use_inspection.
 */
template<typename NodeType>
inline NodeType *
TryGetOwnerNode(const rvsdg::Input & input) noexcept
{
  auto owner = input.GetOwner();
  if (const auto node = std::get_if<Node *>(&owner))
  {
    return dynamic_cast<NodeType *>(*node);
  }
  else
  {
    return nullptr;
  }
}

/**
 * \brief Checks if this is an output to a node of specified type.
 *
 * \tparam NodeType
 *   The node type to be matched against.
 *
 * \param output
 *   Output to be checked.
 *
 * \returns
 *   Owning node of requested type or nullptr.
 *
 * Checks if the specified output belongs to a node of requested type.
 * If this is the case, returns a pointer to the node of matched type.
 * If this is not the case (because either this as a region entry
 * argument or its owning node is not of the requested type), returns
 * nullptr.
 *
 * See \ref def_use_inspection.
 */
template<typename NodeType>
inline NodeType *
TryGetOwnerNode(const rvsdg::Output & output) noexcept
{
  auto owner = output.GetOwner();
  if (const auto node = std::get_if<Node *>(&owner))
  {
    return dynamic_cast<NodeType *>(*node);
  }
  else
  {
    return nullptr;
  }
}

/**
 * \brief Asserts that this is an input to a node of specified type.
 *
 * \tparam NodeType
 *   The node type to be matched against.
 *
 * \param input
 *   Input to be checked.
 *
 * \returns
 *   Owning node of requested type.
 *
 * Checks if the specified input belongs to a node of requested type.
 * If this is the case, returns a reference to the node of matched type,
 * otherwise throws std::logic_error.
 *
 * See \ref def_use_inspection.
 */
template<typename NodeType>
inline NodeType &
AssertGetOwnerNode(const rvsdg::Input & input)
{
  auto node = TryGetOwnerNode<NodeType>(input);
  if (!node)
  {
    throw std::logic_error(std::string("expected node of type ") + typeid(NodeType).name());
  }
  return *node;
}

/**
 * \brief Asserts that this is an output of a node of specified type.
 *
 * \tparam NodeType
 *   The node type to be matched against.
 *
 * \param output
 *   Output to be checked.
 *
 * \returns
 *   Owning node of requested type.
 *
 * Checks if the specified output belongs to a node of requested type.
 * If this is the case, returns a reference to the node of matched type,
 * otherwise throws std::logic_error.
 *
 * See \ref def_use_inspection.
 */
template<typename NodeType>
inline NodeType &
AssertGetOwnerNode(const rvsdg::Output & output)
{
  auto node = TryGetOwnerNode<NodeType>(output);
  if (!node)
  {
    throw std::logic_error(std::string("expected node of type ") + typeid(NodeType).name());
  }
  return *node;
}

/**
 * \brief Checks if the input belongs to a node of the specified operation type.
 *
 * \tparam OperationType
 *   The subclass of Operation to check for.
 *
 * \param input
 *   The output being checked.
 *
 * \returns
 *   True if the input is owned by a node, whose operation is an instance of the specified type.
 *   Otherwise, false.
 */
template<typename OperationType>
bool
IsOwnerNodeOperation(const rvsdg::Input & input) noexcept
{
  return is<OperationType>(TryGetOwnerNode<Node>(input));
}

/**
 * \brief Checks if the output belongs to a node of the specified operation type.
 *
 * \tparam OperationType
 *   The subclass of Operation to check for.
 *
 * \param output
 *   The output being checked.
 *
 * \returns
 *   True if the output is owned by a node, whose operation is an instance of the specified type.
 *   Otherwise, false.
 */
template<typename OperationType>
bool
IsOwnerNodeOperation(const rvsdg::Output & output) noexcept
{
  return is<OperationType>(TryGetOwnerNode<Node>(output));
}

inline Region *
TryGetOwnerRegion(const rvsdg::Input & input) noexcept
{
  auto owner = input.GetOwner();
  if (auto region = std::get_if<Region *>(&owner))
  {
    return *region;
  }
  else
  {
    return nullptr;
  }
}

inline Region *
TryGetOwnerRegion(const rvsdg::Output & output) noexcept
{
  auto owner = output.GetOwner();
  if (auto region = std::get_if<Region *>(&owner))
  {
    return *region;
  }
  else
  {
    return nullptr;
  }
}

static inline std::vector<jlm::rvsdg::Output *>
operands(const Node * node)
{
  std::vector<jlm::rvsdg::Output *> operands;
  for (size_t n = 0; n < node->ninputs(); n++)
    operands.push_back(node->input(n)->origin());
  return operands;
}

static inline std::vector<jlm::rvsdg::Output *>
outputs(const Node * node)
{
  std::vector<jlm::rvsdg::Output *> outputs;
  for (size_t n = 0; n < node->noutputs(); n++)
    outputs.push_back(node->output(n));
  return outputs;
}

/**
 * Returns a subset of the outputs from a node as a vector.
 *
 * @param node The node from which the outputs are taken.
 * @param startIdx The index of the first output.
 * @param size The number of outputs that are returned.
 * @return A vector of outputs.
 *
 * \pre The \p startIdx + \p size must be smaller or equal than the number of outputs of \p node.
 */
static inline std::vector<Output *>
Outputs(const Node & node, const size_t startIdx, const size_t size)
{
  JLM_ASSERT(startIdx + size <= node.noutputs());

  std::vector<Output *> outputs;
  for (size_t n = startIdx; n < startIdx + size; n++)
    outputs.push_back(node.output(n));

  JLM_ASSERT(outputs.size() == size);
  return outputs;
}

static inline void
divert_users(Node * node, const std::vector<Output *> & outputs)
{
  JLM_ASSERT(node->noutputs() == outputs.size());

  for (size_t n = 0; n < outputs.size(); n++)
    node->output(n)->divert_users(outputs[n]);
}

}

#endif

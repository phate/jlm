/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_LAMBDA_HPP
#define JLM_LLVM_IR_OPERATORS_LAMBDA_HPP

#include <jlm/llvm/ir/attribute.hpp>
#include <jlm/llvm/ir/linkage.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/util/iterator_range.hpp>

#include <utility>

namespace jlm::lambda {

/** \brief Lambda operation
*
* A lambda operation determines a lambda's name and \ref FunctionType "function type".
*/
class operation final : public jive::structural_op
{
public:
  ~operation() override;

  operation(
    FunctionType type,
    std::string name,
    const jlm::linkage & linkage,
    attributeset attributes)
    : type_(std::move(type))
    , name_(std::move(name))
    , linkage_(linkage)
    , attributes_(std::move(attributes))
  {}

  operation(const operation & other)
    : type_(other.type_)
    , name_(other.name_)
    , linkage_(other.linkage_)
    , attributes_(other.attributes_)
  {}

  operation(operation && other) noexcept
    : type_(std::move(other.type_))
    , name_(std::move(other.name_))
    , linkage_(other.linkage_)
  {}

  operation &
  operator=(const operation & other)
  {
    if (this == &other)
      return *this;

    type_ = other.type_;
    name_ = other.name_;
    linkage_ = other.linkage_;
    attributes_ = other.attributes_;

    return *this;
  }

  operation &
  operator=(operation && other) noexcept
  {
    if (this == &other)
      return *this;

    type_ = std::move(other.type_);
    name_ = std::move(other.name_);
    linkage_ = other.linkage_;
    attributes_ = std::move(other.attributes_);

    return *this;
  }

  [[nodiscard]] const FunctionType &
  type() const noexcept
  {
    return type_;
  }

  [[nodiscard]] const std::string &
  name() const noexcept
  {
    return name_;
  }

  [[nodiscard]] const jlm::linkage &
  linkage() const noexcept
  {
    return linkage_;
  }

  [[nodiscard]] const attributeset &
  attributes() const noexcept
  {
    return attributes_;
  }

  [[nodiscard]] std::string
  debug_string() const override;

  bool
  operator==(const jive::operation & other) const noexcept override;

  [[nodiscard]] std::unique_ptr<jive::operation>
  copy() const override;

private:
  FunctionType type_;
  std::string name_;
  jlm::linkage linkage_;
  attributeset attributes_;
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
class node final : public jive::structural_node
{
  class cviterator;
  class cvconstiterator;

  class fctargiterator;
  class fctargconstiterator;

  class fctresiterator;
  class fctresconstiterator;

  using fctargument_range = iterator_range<fctargiterator>;
  using fctargument_constrange = iterator_range<fctargconstiterator>;

  using ctxvar_range = iterator_range<cviterator>;
  using ctxvar_constrange = iterator_range<cvconstiterator>;

  using fctresult_range = iterator_range<fctresiterator>;
  using fctresult_constrange = iterator_range<fctresconstiterator>;

public:
  ~node() override;

private:
  node(
    jive::region * parent,
    lambda::operation && op)
    : structural_node(op, parent, 1)
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

  [[nodiscard]] jive::region *
  subregion() const noexcept
  {
    return structural_node::subregion(0);
  }

  [[nodiscard]] const lambda::operation &
  operation() const noexcept
  {
    return *AssertedCast<const lambda::operation>(&structural_node::operation());
  }

  [[nodiscard]] const FunctionType &
  type() const noexcept
  {
    return operation().type();
  }

  [[nodiscard]] const std::string &
  name() const noexcept
  {
    return operation().name();
  }

  [[nodiscard]] const jlm::linkage &
  linkage() const noexcept
  {
    return operation().linkage();
  }

  [[nodiscard]] const attributeset &
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
  add_ctxvar(jive::output * origin);

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
  copy(
    jive::region * region,
    const std::vector<jive::output*> & operands) const override;

  lambda::node *
  copy(
    jive::region * region,
    jive::substitution_map & smap) const override;

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
    jive::region * parent,
    const FunctionType & type,
    const std::string & name,
    const jlm::linkage & linkage,
    const attributeset & attributes);

  /**
  * See \ref create().
  */
  static node *
  create(
    jive::region * parent,
    const FunctionType & type,
    const std::string & name,
    const jlm::linkage & linkage)
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
  finalize(const std::vector<jive::output*> & results);

  /**
  * Retrieves all direct calls of a lambda node.
  *
  * \param calls A vector for the direct calls. If vector is NULL, then no call will be
  * retrieved.
  *
  * \return True if the lambda has only direct calls, otherwise False.
  */
  bool
  direct_calls(std::vector<jive::simple_node*> * calls = nullptr) const;
};

/** \brief Lambda context variable input
*/
class cvinput final : public jive::structural_input
{
  friend ::jlm::lambda::node;

public:
  ~cvinput() override;

private:
  cvinput(
    lambda::node * node,
    jive::output * origin)
    : structural_input(node, origin, origin->port())
  {}

  static cvinput *
  create(
    lambda::node * node,
    jive::output * origin)
  {
    auto input = std::unique_ptr<cvinput>(new cvinput(node, origin));
    return AssertedCast<cvinput>(node->append_input(std::move(input)));
  }

public:
  [[nodiscard]] cvargument *
  argument() const noexcept;

  [[nodiscard]] lambda::node *
  node() const noexcept
  {
    return AssertedCast<lambda::node>(structural_input::node());
  }
};

/** \brief Lambda context variable iterator
*/
class node::cviterator final : public jive::input::iterator<cvinput>
{
  friend ::jlm::lambda::node;

  constexpr explicit
  cviterator(cvinput * input)
    : jive::input::iterator<cvinput>(input)
  {}

  [[nodiscard]] cvinput *
  next() const override
  {
    auto node = value()->node();
    auto index = value()->index();

    return node->ninputs() > index+1 ? node->input(index+1) : nullptr;
  }
};

/** \brief Lambda context variable const iterator
*/
class node::cvconstiterator final : public jive::input::constiterator<cvinput>
{
  friend ::jlm::lambda::node;

  constexpr explicit
  cvconstiterator(const cvinput * input)
    : jive::input::constiterator<cvinput>(input)
  {}

  [[nodiscard]] const cvinput *
  next() const override
  {
    auto node = value()->node();
    auto index = value()->index();

    return node->ninputs() > index+1 ? node->input(index+1) : nullptr;
  }
};

/** \brief Lambda output
*/
class output final : public jive::structural_output
{
  friend ::jlm::lambda::node;

public:
  ~output() override;

private:
  output(
    lambda::node * node,
    const jive::port & port)
    : structural_output(node, port)
  {}

  static output *
  create(
    lambda::node * node,
    const jive::port & port)
  {
    auto output = std::unique_ptr<lambda::output>(new lambda::output(node, port));
    return AssertedCast<lambda::output>(node->append_output(std::move(output)));
  }

public:
  lambda::node *
  node() const noexcept
  {
    return AssertedCast<lambda::node>(structural_output::node());
  }
};

/** \brief Lambda function argument
*/
class fctargument final : public jive::argument
{
  friend ::jlm::lambda::node;

public:
  ~fctargument() override;

  const attributeset &
  attributes() const noexcept
  {
    return attributes_;
  }

  void
  add(const jlm::attribute & attribute)
  {
    attributes_.insert(attribute);
  }

  void
  set_attributes(const attributeset & attributes)
  {
    attributes_ = attributes;
  }

private:
  fctargument(
    jive::region * region,
    const jive::type & type)
    : jive::argument(region, nullptr, type)
  {}

  static fctargument *
  create(
    jive::region * region,
    const jive::type & type)
  {
    auto argument = new fctargument(region, type);
    region->append_argument(argument);
    return argument;
  }

  attributeset attributes_;
};

/** \brief Lambda function argument iterator
*/
class node::fctargiterator final : public jive::output::iterator<lambda::fctargument>
{
  friend ::jlm::lambda::node;

  constexpr explicit
  fctargiterator(lambda::fctargument * argument)
    : jive::output::iterator<lambda::fctargument>(argument)
  {}

  [[nodiscard]] lambda::fctargument *
  next() const override
  {
    auto index = value()->index();
    auto lambda = AssertedCast<lambda::node>(value()->region()->node());

    /*
      This assumes that all function arguments were added to the lambda region
      before any context variable was added.
    */
    return lambda->nfctarguments() > index+1
           ? lambda->fctargument(index+1)
           : nullptr;
  }
};

/** \brief Lambda function argument const iterator
*/
class node::fctargconstiterator final : public jive::output::constiterator<lambda::fctargument>
{
  friend ::jlm::lambda::node;

  constexpr explicit
  fctargconstiterator(const lambda::fctargument * argument)
    : jive::output::constiterator<lambda::fctargument>(argument)
  {}

  [[nodiscard]] const lambda::fctargument *
  next() const override
  {
    auto index = value()->index();
    auto lambda = AssertedCast<lambda::node>(value()->region()->node());

    /*
      This assumes that all function arguments were added to the lambda region
      before any context variable was added.
    */
    return lambda->nfctarguments() > index+1
           ? lambda->fctargument(index+1)
           : nullptr;
  }
};

/** \brief Lambda context variable argument
*/
class cvargument final : public jive::argument
{
  friend ::jlm::lambda::node;

public:
  ~cvargument() override;

private:
  cvargument(
    jive::region * region,
    cvinput * input)
    : jive::argument(region, input, input->port())
  {}

  static cvargument *
  create(
    jive::region * region,
    lambda::cvinput * input)
  {
    auto argument = new cvargument(region, input);
    region->append_argument(argument);
    return argument;
  }

public:
  cvinput *
  input() const noexcept
  {
    return AssertedCast<cvinput>(jive::argument::input());
  }
};

/** \brief Lambda result
*/
class result final : public jive::result
{
  friend ::jlm::lambda::node;

public:
  ~result() override;

private:
  explicit
  result(jive::output * origin)
    : jive::result(origin->region(), origin, nullptr, origin->port())
  {}

  static result *
  create(jive::output * origin)
  {
    auto result = new lambda::result(origin);
    origin->region()->append_result(result);
    return result;
  }

public:
  lambda::output *
  output() const noexcept
  {
    return AssertedCast<lambda::output>(jive::result::output());
  }
};

/** \brief Lambda result iterator
*/
class node::fctresiterator final : public jive::input::iterator<lambda::result>
{
  friend ::jlm::lambda::node;

  constexpr explicit
  fctresiterator(lambda::result * result)
    : jive::input::iterator<lambda::result>(result)
  {}

  [[nodiscard]] lambda::result *
  next() const override
  {
    auto index = value()->index();
    auto lambda = AssertedCast<lambda::node>(value()->region()->node());

    return lambda->nfctresults() > index+1
           ? lambda->fctresult(index+1)
           : nullptr;
  }
};

/** \brief Lambda result const iterator
*/
class node::fctresconstiterator final : public jive::input::constiterator<lambda::result>
{
  friend ::jlm::lambda::node;

  constexpr explicit
  fctresconstiterator(const lambda::result * result)
    : jive::input::constiterator<lambda::result>(result)
  {}

  [[nodiscard]] const lambda::result *
  next() const override
  {
    auto index = value()->index();
    auto lambda = AssertedCast<lambda::node>(value()->region()->node());

    return lambda->nfctresults() > index+1
           ? lambda->fctresult(index+1)
           : nullptr;
  }
};

}

static inline bool
is_exported(const jlm::lambda::node & lambda)
{
  for (auto & user : *lambda.output()) {
    if (dynamic_cast<const jive::expport*>(&user->port()))
      return true;
  }

  return false;
}

#endif

/*
 * Copyright 2012 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2012 2013 2014 2025 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_PHI_HPP
#define JLM_RVSDG_PHI_HPP

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/util/common.hpp>

namespace jlm::rvsdg
{

class PhiOperation final : public rvsdg::StructuralOperation
{
public:
  ~PhiOperation() override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;
};

class PhiBuilder;

/**
 * \brief A phi node represents the fixpoint of mutually recursive definitions.
 *
 * A phi node may contain multiple definitions of objects that mutually
 * reference each other. It represents the fixpoint of infinitely unrolling
 * the definitions.
 *
 * A phi node may reference external definitions as \ref ContextVar "ContextVar"s,
 * and it may both define and reference several mutually recursive
 * objects as \ref FixVar "FixVar"s.
 */
class PhiNode final : public rvsdg::StructuralNode
{
  friend class PhiBuilder;

public:
  ~PhiNode() override;

private:
  explicit PhiNode(rvsdg::Region * parent)
      : StructuralNode(parent, 1)
  {}

  static PhiNode *
  create(rvsdg::Region * parent)
  {
    return new PhiNode(parent);
  }

public:
  /**
   * \brief Bound context variable
   *
   * Context variables represent external definitions made available
   * to a phi construct. These are realized as inputs to the
   * phi node itself, and made accessible to the body of the
   * phi construct in the form of external subregion arguments.
   */
  struct ContextVar
  {
    /**
     * \brief Input variable bound into the phi construct.
     *
     * The input port into the phi node that supplies the value
     * of the context variable bound into the phi construct.
     */
    rvsdg::Input * input;

    /**
     * \brief Access to bound object in subregion.
     *
     * Supplies access to the value bound into the phi construct
     * from inside the region contained in the phi node. This
     * evaluates to the value bound into the phi.
     */
    rvsdg::Output * inner;
  };

  /**
   * \brief Description of a recursively defined variable.
   *
   * A recursively defined variable from the POV of a phi node
   * has multiple representations. This structure bundles
   * all representations into a single object.
   */
  struct FixVar
  {
    /**
     * \brief Reference to mutual-recursively defined object in phi.
     *
     * This is an argument slot in the phi region and represents a reference to an
     * object defined within the phi. Definitions in the phi region
     * that want to refer to another object in the phi region must use this
     * handle to refer to it (this can be either an object referring to itself
     * or referring to another mutually recursive object).
     */
    rvsdg::Output * recref;

    /**
     * \brief Definition result of a variable within the phi region.
     *
     * This is a result slot in the phi region and represents the finalized
     * definition of a variable with the phi region.
     */
    rvsdg::Input * result;

    /**
     * \brief Output of phi region representing externally available definition.
     *
     * This output of the phi node allows to externally refer to the objects
     * defined in the phi.
     */
    rvsdg::Output * output;
  };

  [[nodiscard]] const PhiOperation &
  GetOperation() const noexcept override;

  /**
   * \brief Adds a context variable to the phi node.
   *
   * \param origin
   *   The value to be bound into the phi node.
   *
   * \pre
   *   \p origin must be from the same region as the phi node.
   *
   * \return The context variable argument of the phi abstraction.
   *
   * Binds a new variable as context variable into the phi node.
   * Its value can now be referenced from within the phi region.
   */
  ContextVar
  AddContextVar(jlm::rvsdg::Output & origin);

  /**
   * \brief Removes context variables from phi node
   *
   * \param vars
   *   Variables to be removed.
   *
   * \pre
   *   \p vars must be context variables of the phi node
   *   and they must be unused within the region.
   *
   * Removes the given context variables.
   */
  void
  RemoveContextVars(std::vector<ContextVar> vars);

  /**
   * \brief Gets all bound context variables.
   *
   * \returns
   *   The context variable descriptions.
   *
   * Returns all context variable descriptions.
   */
  [[nodiscard]] std::vector<ContextVar>
  GetContextVars() const noexcept;

  /**
   * \brief Maps input to context variable.
   *
   * \param input
   *   Input to the phi node.
   *
   * \returns
   *   The context variable description corresponding to the input.
   *
   * \pre
   *   \p input must be input to this node.
   *
   * Returns the context variable description corresponding
   * to this input of the phi node. All inputs to the phi
   * node are by definition bound context variables that are
   * accessible in the subregion through the corresponding
   * argument.
   */
  [[nodiscard]] ContextVar
  MapInputContextVar(const rvsdg::Input & input) const noexcept;

  /**
   * \brief Attempts to map bound variable reference to context variable.
   *
   * \param argument
   *   Region argument to phi subregion.
   *
   * \returns
   *   The context variable description corresponding to \p argument,
   *   or \p std::nullopt.
   *
   * \pre
   *   \p argument must be an argument to the subregion of this node.
   *
   * Checks whether the given argument corresponds to a context
   * variable and returns its description in that case. All
   * arguments of a phi region are either context or fixpoint
   * variables, see \ref MapArgumentFixVar.
   */
  [[nodiscard]] std::optional<ContextVar>
  MapArgumentContextVar(const rvsdg::Output & argument) const noexcept;

  /**
   * \brief Removes fixpoint variables from the phi node
   *
   * \param vars
   *   Variables to be removed.
   *
   * \pre
   *   \p vars must be fixpoint variables of the phi node,
   *   and they must be identity mappings within the phi region.
   *
   * Removes the given fixpoint variables
   */
  void
  RemoveFixVars(std::vector<FixVar> vars);

  /**
   * \brief Gets all fixpoint variables.
   *
   * \returns
   *   The fixpoint variable descriptions.
   *
   * Returns all fixpoint variable descriptions.
   */
  [[nodiscard]] std::vector<FixVar>
  GetFixVars() const noexcept;

  /**
   * \brief Tries to map region argument to fixpoint variable.
   *
   * \param argument
   *   The argument of the region to be mapped.
   *
   * \returns
   *   Fixpoint variable description corresponding to \p argument,
   *   or \p std::nullopt.
   *
   * \pre
   *   \p argument must be an argument of the subregion contained in
   *   the phi node.
   *
   * Checks whether the given argument corresponds to a fixpoint
   * variable and returns its description in that case. All
   * arguments of a phi region are either context or fixpoint
   * variables, see \ref MapArgumentContextVar.
   */
  [[nodiscard]] std::optional<FixVar>
  MapArgumentFixVar(const rvsdg::Output & argument) const noexcept;

  /**
   * \brief Maps region result to fixpoint variable.
   *
   * \param result
   *   The result of the region to be mapped.
   *
   * \returns
   *   Fixpoint variable description corresponding to \p input.
   *
   * \pre
   *   \p result must be a result of the subregion contained in
   *   the phi node.
   *
   * Maps result of the region to a fixpoint variable.
   */
  [[nodiscard]] FixVar
  MapResultFixVar(const rvsdg::Input & result) const noexcept;

  /**
   * \brief Maps output to fixpoint variable.
   *
   * \param output
   *   The output of the phi node to be mapped.
   *
   * \returns
   *   Fixpoint variable description corresponding to \p output.
   *
   * \pre
   *   \p output must be an output of this phi node.
   *
   * Maps output of the phi node to a fixpoint variable.
   */
  [[nodiscard]] FixVar
  MapOutputFixVar(const rvsdg::Output & output) const noexcept;

  /**
   * \brief Maps region argument to its function.
   *
   * \param argument
   *   The argument of the phi node to be mapped.
   *
   * \returns
   *   Either the fixpoint variable description or the
   *   context variable description, depending on the
   *   designation of the argument.
   *
   * \pre
   *   \p argument must be an argument of the region within this phi node.
   *
   * Maps output of the phi node to either a fixpoint or context variable.
   */
  [[nodiscard]] std::variant<FixVar, ContextVar>
  MapArgument(const rvsdg::Output & argument) const noexcept;

  rvsdg::Region *
  subregion() const noexcept
  {
    return StructuralNode::subregion(0);
  }

  PhiNode *
  copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const override;

  /**
   * Extracts all lambda nodes from a phi node.
   *
   * The function is capable of handling nested phi nodes.
   *
   * @param phiNode The phi node from which the lambda nodes should be extracted.
   * @return A vector of lambda nodes.
   */
  static std::vector<rvsdg::LambdaNode *>
  ExtractLambdaNodes(const PhiNode & phiNode);
};

/** Helper class to incrementally construct a well-formed phi object. */
class PhiBuilder final
{
public:
  constexpr PhiBuilder() noexcept
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

    node_ = PhiNode::create(parent);
  }

  PhiNode::ContextVar
  AddContextVar(jlm::rvsdg::Output & origin);

  PhiNode::FixVar
  AddFixVar(std::shared_ptr<const jlm::rvsdg::Type> type);

  PhiNode *
  end();

private:
  PhiNode * node_;
};

}

#endif

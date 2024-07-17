/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_IR_STATIC_LOOP_HPP
#define JLM_HLS_IR_STATIC_LOOP_HPP

//FIXME check what's needed
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/operation.hpp>
#include <jlm/rvsdg/structural-node.hpp> // needed
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <jlm/hls/ir/static-hls.hpp>
#include <jlm/hls/ir/static/fsm.hpp>

namespace jlm::static_hls
{

/*! \class loop_op
* \brief Loop operation for static loop node.
* See loop_node for more details.
*/
class loop_op final : public jlm::rvsdg::structural_op
{
public:
  virtual ~loop_op() noexcept
  {}

  std::string
  debug_string() const override
  {
    return "SHLS_LOOP";
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new loop_op(*this));
  }
};

class backedge_argument;
class backedge_result;

/*! \class loop_node
    \brief Static loop node for HLS.

    Theta node are lowered into this node.
    This node has 2 regions :
      - control region : contains the control flow of the loop, registers, muxes and FSM
      - compute region : contains the operations that can be computed in parallel
*/
class loop_node final : public jlm::rvsdg::structural_node
{
// public:
//   ~loop_node();

private:
  inline loop_node(jlm::rvsdg::region * parent)
      : structural_node(loop_op(), parent, 2)
  {}

public:
  //FIXME add a method that removes muxes with only one input

  /*! \brief Debug function that prints the registers for each node output of the control region.
  */
  void print_nodes_registers() const;

  //TODO doc
  void
  print_fsm() const;

  /*! \brief Gets the connected control region result from a node input in the compute region.
  * \param input The node input in the compute region.
  * \return The connected control region result.
  */
  jlm::rvsdg::result *
  get_origin_result(jlm::rvsdg::node_input * input) const;

  /*! \brief Gets list a register nodes connected to a node input in the compute region.
  * Also prints is the input is a loop input
  * \param node The node input in the compute region.
  * \return The list of register nodes connected to the node input.
  */
  std::vector<jlm::rvsdg::output *>
  get_users(jlm::rvsdg::node_input * node) const;

  /*! \brief get the connected mux to a node input in the compute region.
  */
  jlm::rvsdg::node*
  get_mux(jlm::rvsdg::node_input * node) const;

  /*! \brief Creates a new loop node.
  * Simply calls the constructor.
  * \param parent The parent region of the loop.
  * \return The newly created loop node.
  */
  static loop_node *
  create(jlm::rvsdg::region * parent);

  //FIXME: this doc
  /*! \brief Adds a loop input to the loop.
  * \param origin The origin of the loop input.
  * \return The newly created loop input.
  */
  jlm::rvsdg::structural_output *
  add_loopvar(jlm::rvsdg::theta_input * theta_input);
  
  /*! \brief Returns the compute region of the loop.
  */
  inline jlm::rvsdg::region *
  control_subregion() const noexcept
  {
    return structural_node::subregion(0);
  }

  /*! \brief Returns the compute region of the loop.
  */
  inline jlm::rvsdg::region *
  compute_subregion() const noexcept
  {
    return structural_node::subregion(1);
  }

  /*! \brief Copies the loop node.
  * \param region The parent region of the new loop node.
  * \param smap The substitution map for nodes in the loop subregions.
  * \return The newly created loop node.
  */
  virtual loop_node *
  copy(jlm::rvsdg::region * region, jlm::rvsdg::substitution_map & smap) const override;

  /*! \brief Adds a node to the loop.
  * Used when building the loop_node from a theta node.
  * This method adds a node to the loop_node either by adding an operation to the compute region or by using an already implemented one.
  * Also adds a register in the control region for each of its outputs.
  * This will also add inputs to the muxes in the control region to route the node node.
  * \param node The node to add.
  */
  void
  add_node(jlm::rvsdg::node * node);

  //TODO doc
  void
  add_loopback_arg(jlm::rvsdg::theta_input * theta_input);
  
  /*! \brief Determines if an operation is aleardy added in the compute region.
  * This goes through every node in the compute region to check if the operation is already implemented.
  * \param op The operation to check.
  * \return The node that implements the operation if it is already implemented, nullptr otherwise.
  */
  jlm::rvsdg::node *
  is_op_implemented(const jlm::rvsdg::operation& op) const noexcept;

  /*! \brief Adds a backedge to the loop.
  * \param origin The origin of the backedge.
  * \return The newly created backedge.
  */
  backedge_result *
  add_backedge(jlm::rvsdg::output* origin);

  /*! \brief Replace muxes with only one input by the input node.
  * This is called in the finalize method.
  */
  void
  remove_single_input_muxes();

  //TODO doc
  void
  connect_muxes();

  /*! \brief Finishes the loop building and creates the FSM gamma node
  * This calls remove_single_input_muxes.
  */
  void
  finalize();

private:
  jlm::static_hls::fsm_node_builder * fsm_; // The FSM node of the loop
  //TODO rename
  jlm::rvsdg::substitution_map reg_smap_; // Maps the original node ouput with its substitute register output, also maps arguments
};

//TODO rename those as they are not only used for backedges
class backedge_argument : public jlm::rvsdg::argument
{
  friend loop_node;
  friend backedge_result;

public:
  ~backedge_argument() override = default;

  backedge_result *
  result()
  {
    return result_;
  }

private:
  backedge_argument(jlm::rvsdg::region * region, const std::shared_ptr<const jlm::rvsdg::type> & type)
      : jlm::rvsdg::argument(region, nullptr, type),
        result_(nullptr)
  {}

  static backedge_argument *
  create(jlm::rvsdg::region * region, const std::shared_ptr<const jlm::rvsdg::type> & type)
  {
    auto argument = new backedge_argument(region, type);
    region->append_argument(argument);
    return argument;
  }

  backedge_result * result_;
};

class backedge_result : public jlm::rvsdg::result
{
  friend loop_node;
  friend backedge_argument;

public:
  ~backedge_result() override = default;

  backedge_argument *
  argument() const
  {
    return argument_;
  }

private:
  backedge_result(jlm::rvsdg::output * origin)
      : jlm::rvsdg::result(origin->region(), origin, nullptr, origin->port()),
        argument_(nullptr)
  {}

  static backedge_result *
  create(jlm::rvsdg::output * origin)
  {
    auto result = new backedge_result(origin);
    origin->region()->append_result(result);
    return result;
  }

  backedge_argument * argument_;
};

} // namespace jlm::static_hls

#endif // JLM_HLS_IR_STATIC_LOOP_HPP
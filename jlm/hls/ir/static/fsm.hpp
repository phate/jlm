/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_IR_STATIC_FSM_HPP
#define JLM_HLS_IR_STATIC_FSM_HPP

#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/hls/ir/static-hls.hpp>
#include <jlm/rvsdg/gamma.hpp>

namespace jlm::static_hls
{
class fsm_node_temp;
class fsm_node_builder;

class fsm_state final : private jlm::rvsdg::region
{
  friend fsm_node_temp;
private:
  inline fsm_state(jlm::rvsdg::structural_node * node, size_t index)
      : region(node, index)/*, index_(index)*/
  {}

public:
  using region::copy;

  static fsm_state *
  create(fsm_node_temp * parent_fsm_node, size_t index);

  void
  enable_reg(jlm::rvsdg::node* node);

  //TODO doc
  void
  add_ctl_result(size_t nalternatives, jlm::rvsdg::structural_output* structural_output);

  void
  set_mux_ctl(jlm::rvsdg::input* result, size_t alternatives);

  void apply_mux_ctl();

private:
  // size_t index_;
  std::unordered_map<jlm::rvsdg::input*, size_t> muxes_ctl_;
};

/*! \class fsm_op
* \brief Operation for the Finite State Machine.
* See fsm_node for more details.
*/
class fsm_op final : public jlm::rvsdg::structural_op
{
public:
  virtual ~fsm_op() noexcept
  {}

  std::string
  debug_string() const override
  {
    return "SHLS_FSM";
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new fsm_op(*this));
  }
};

class fsm_node_temp final : public jlm::rvsdg::structural_node
{
  friend fsm_node_builder;
public:
  ~fsm_node_temp();

private:
  inline fsm_node_temp(jlm::rvsdg::region * parent)
    : structural_node(fsm_op(), parent, 1)
  {}

public:
  /*! \brief Copies the loop node.
  * \param region The parent region of the new loop node.
  * \param smap The substitution map for nodes in the loop subregions.
  * \return The newly created loop node.
  */
  virtual fsm_node_temp *
  copy(jlm::rvsdg::region * region, jlm::rvsdg::substitution_map & smap) const override;

  void
  print_states() const;

  //TODO finish this doc
  /*! \brief Connects a register store signal input to the fsm.
  */
  jlm::rvsdg::structural_output *
  add_register_ouput()
  {
    return add_ctl_output(2);
  };

  jlm::rvsdg::structural_output *
  add_ctl_output(size_t nalternatives);

  // //TODO finish this doc
  // /*! \brief Connects a register store signal input to the fsm.
  // */
  // jlm::rvsdg::structural_output *
  // add_mux_ouput();

  /*! \brief Adds a new state to the fsm.
  * Creates a fsm_state instance that is added to the states_ vector
  */
  fsm_state *
  add_state();

private:
  std::vector<fsm_state*> states_;
};

/*! \class fsm_node
    \brief Finite State Machine node for HLS.
*/
class fsm_node_builder final
{
public:
  ~fsm_node_builder();

private:
  inline fsm_node_builder(jlm::rvsdg::region * parent)
  {
    fsm_node_temp_ = new fsm_node_temp(parent);
  };

public:
  /*! \brief Creates a new fsm node.
  * Simply calls the constructor.
  * \param parent The parent region of the fsm.
  * \return The newly created fsm node.
  */
  static fsm_node_builder *
  create(jlm::rvsdg::region * parent);

  // /*! \brief Copies the loop node.
  // * \param region The parent region of the new loop node.
  // * \param smap The substitution map for nodes in the loop subregions.
  // * \return The newly created loop node.
  // */
  // virtual fsm_node *
  // copy(jlm::rvsdg::region * region, jlm::rvsdg::substitution_map & smap) const override;

  //TODO finish this doc
  /*! \brief Connects a register store signal input to the fsm.
  */
  jlm::rvsdg::structural_output *
  add_register_ouput() {
    return fsm_node_temp_->add_register_ouput();
  };

  //TODO finish this doc
  /*! \brief Connects a register store signal input to the fsm.
  */
  jlm::rvsdg::structural_output *
  add_ctl_output(size_t nalternatives) {
    return fsm_node_temp_->add_ctl_output(nalternatives);
  };

  // //TODO finish this doc
  // /*! \brief Connects a register store signal input to the fsm.
  // */
  // jlm::rvsdg::structural_output *
  // add_mux_ouput() {
  //   return fsm_node_temp_->add_mux_ouput();
  // };

  /*! \brief Adds a new state to the fsm.
  * Creates a fsm_state instance that is added to the states_ vector
  */
  fsm_state *
  add_state() {
    return fsm_node_temp_->add_state();
  };

  void
  apply_mux_ctl()
  {
    for (auto state : fsm_node_temp_->states_) state->apply_mux_ctl();
  }

  /*! \brief Generates the gamma node (which is the final implementation) for the fsm.
  */
  void
  generate_gamma(jlm::rvsdg::output *predicate);

  inline size_t
  nalternatives() const
  {
    if (!gamma_) return fsm_node_temp_->states_.size();
    return gamma_->nsubregions();
  }

private:
  fsm_node_temp* fsm_node_temp_ = nullptr;
  jlm::rvsdg::gamma_node* gamma_ = nullptr;
};

} // namespace jlm::static_hls

#endif // JLM_HLS_IR_STATIC_FSM_HPP
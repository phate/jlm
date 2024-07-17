/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_IR_STATIC_HLS_HPP
#define JLM_HLS_IR_STATIC_HLS_HPP

//FIXME check what's needed
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/operation.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/theta.hpp>

namespace jlm::static_hls
{

// class mux_ctltype final : public jlm::rvsdg::statetype
// {
// public:
//   virtual ~mux_ctltype() noexcept;

//   virtual std::string
//   debug_string() const override;

//   virtual bool
//   operator==(const jlm::rvsdg::type & other) const noexcept override;

//   std::shared_ptr<const jlm::rvsdg::type>
//   copy() const override;

//   inline size_t
//   nalternatives() const noexcept
//   {
//     return nalternatives_;
//   }

//   /**
//    * \brief Instantiates control type
//    *
//    * \returns Control type instance
//    *
//    * Creates an instance of a control type capable of representing
//    * the specified number of alternatives. The returned instance
//    * will usually be a static singleton for the type.
//    */
//   static std::shared_ptr<const mux_ctltype>
//   Create();

// private:
//   size_t nalternatives_;
// };

class mux_op final : public jlm::rvsdg::simple_op
{
private:
  mux_op(std::vector<std::shared_ptr<const jlm::rvsdg::type>>& operands_type, const std::shared_ptr<const jlm::rvsdg::type>& result_type)
      // : jlm::rvsdg::simple_op(create_portvector(nalternatives, type), { type })
      : jlm::rvsdg::simple_op(operands_type, { result_type })
  {}

public:
  virtual ~mux_op()
  {}

  inline size_t nalternatives() const{ return nalternatives_; }
  inline size_t has_predicate() const{ return has_predicate_; }

  std::string
  debug_string() const override
  {
    return "SHLS_MUX";
  };

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    auto ot = dynamic_cast<const mux_op *>(&other);
    // check predicate and value
    return ot && ot->argument(0).type() == argument(0).type()
        && ot->result(0).type() == result(0).type();
  };

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new mux_op(*this));
  };

  static jlm::rvsdg::simple_node*
  create(
      jlm::rvsdg::output & predicate,
      const std::vector<jlm::rvsdg::output *> & alternatives)
  {
    if (alternatives.empty())
      throw util::error("Insufficient number of operands.");
    auto ctl = dynamic_cast<const jlm::rvsdg::ctltype *>(&predicate.type());
    if (!ctl)
      throw util::error("Predicate needs to be a ctltype.");
    if (alternatives.size() != ctl->nalternatives())
      throw util::error("Alternatives and predicate do not match.");
    
    auto region = predicate.region();
    auto operands = std::vector<jlm::rvsdg::output *>();
    operands.push_back(&predicate);
    operands.insert(operands.end(), alternatives.begin(), alternatives.end());
    auto operands_type = std::vector<std::shared_ptr<const jlm::rvsdg::type>>(alternatives.size() + 1, alternatives.front()->Type());
    operands_type.at(0) = jlm::rvsdg::ctltype::Create(alternatives.size());
    mux_op op(operands_type, alternatives.front()->Type());
    op.nalternatives_ = alternatives.size();
    op.has_predicate_ = true;
    return jlm::rvsdg::simple_node::create(region, op, operands);
  };

  static jlm::rvsdg::simple_node*
  create(
      const std::vector<jlm::rvsdg::output *> & alternatives)
  {
    if (alternatives.empty())
      throw util::error("Insufficient number of operands.");
    
    auto region = alternatives[0]->region();
    auto operands = std::vector<jlm::rvsdg::output *>();

    // operands.push_back();
    operands.insert(operands.end(), alternatives.begin(), alternatives.end());
    auto operands_type = std::vector<std::shared_ptr<const jlm::rvsdg::type>>(alternatives.size(), alternatives.front()->Type());
    mux_op op(operands_type, alternatives.front()->Type());
    op.nalternatives_ = alternatives.size();
    op.has_predicate_ = false;
    return jlm::rvsdg::simple_node::create(region, op, operands);
  };

private:
  // static std::vector<jlm::rvsdg::port>
  // create_portvector(size_t nalternatives, const jlm::rvsdg::type & type)
  // {
  //   auto vec = std::vector<jlm::rvsdg::port>(nalternatives + 1, type);
  //   vec[0] = jlm::rvsdg::ctltype(nalternatives);
  //   return vec;
  // };

  size_t nalternatives_;
  bool has_predicate_;
};

//TODO doc
/*! \brief Adds a input to a mux node.
* Internally create a new node and remove the old one !!
* \param old_mux The mux node to add the input to.
* \param new_input The input to add to the mux.
* \return The new mux node with the added new input.
*/
jlm::rvsdg::node*
mux_add_input(jlm::rvsdg::node* old_mux, jlm::rvsdg::output* new_input, bool predicate=false);


//TODO doc
inline jlm::rvsdg::node*
mux_connect_predicate(jlm::rvsdg::node* old_mux, jlm::rvsdg::output* predicate)
{
  return mux_add_input(old_mux, predicate, true);
};

/*! \brief A register operation with a store predicate, data input and data output.
*/
extern size_t instances_count;
class reg_op final : public jlm::rvsdg::simple_op
{
public:
  virtual ~reg_op()
  {}

  reg_op(const std::shared_ptr<const jlm::rvsdg::type> & type)
      : jlm::rvsdg::simple_op(std::vector<std::shared_ptr<const jlm::rvsdg::type>>{jlm::rvsdg::ctltype::Create(2), type}, { type })
      // : jlm::rvsdg::simple_op(std::vector<jlm::rvsdg::port>{type}, { type })
  {
    id_ = instances_count++;
  }

  std::string
  debug_string() const override
  {
    if (origin_debug_string_.empty()) return jlm::util::strfmt("SHLS_REG", id_);
    return jlm::util::strfmt("SHLS_REG", id_, "(", origin_debug_string_, ")");
  }

  bool
  operator==(const jlm::rvsdg::operation & other) const noexcept override
  {
    return true; //TODO check if that's how to do it
  }

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override
  {
    return std::unique_ptr<jlm::rvsdg::operation>(new reg_op(*this));
  }

  /*! \brief Creates a new register node.
  * \param store_input The store predicate of the register.
  * \param input The data input of the register.
  * \return The newly created register node.
  */
  static jlm::rvsdg::node *
  create(
      jlm::rvsdg::output & store_input, 
      jlm::rvsdg::output & input,
      std::string origin_debug_string)
  {
    reg_op op(input.Type());
    op.origin_debug_string_ = origin_debug_string;
    return jlm::rvsdg::simple_node::create(input.region(), op, { &store_input, &input });
    // return jlm::rvsdg::simple_node::create_normalized(input.region(), op, { &store_input, &input })[0];
  }

private:
  // static std::vector<jlm::rvsdg::port>
  // create_portvector(size_t nalternatives, const jlm::rvsdg::type & type)
  // {
  //   auto vec = std::vector<jlm::rvsdg::port>(nalternatives + 1, type);
  //   vec[0] = jlm::rvsdg::ctltype(nalternatives);
  //   return vec;
  // }
  size_t id_ = 0; //TODO delete this, only for debugging
  std::string origin_debug_string_ = "";
};

} // namespace jlm::static_hls

#endif // JLM_HLS_IR_STATIC_HLS_HPP